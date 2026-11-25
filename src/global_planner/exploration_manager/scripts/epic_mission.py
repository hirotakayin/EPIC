#!/usr/bin/env python3
"""Flies EPIC through the same mission as the CERLAB presets: same start, takeoff, entry waypoints and return home.

EPIC only explores, so this node owns the controller's command topic. It flies takeoff, entry, return and landing
itself and forwards traj_server's commands while exploring, limiting yaw rate on everything it sends.
"""
import heapq
import json
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from quadrotor_msgs.msg import PositionCommand
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger, TriggerResponse
from traj_utils.msg import PolyTraj
from visualization_msgs.msg import Marker

RATE = 30.0
CRUMB_SPACING = 0.25 # m between remembered positions on the flown path
CRUMB_LINK = 0.8 # m, EPIC keeps 0.45 m from obstacles, so remembered positions this close have no beam (0.4 m) between them
GOAL_MOVED = 1.0 # m, EPIC replans all the time: a new plan row only when the goal has moved this far
CMD_TIMEOUT = 0.5 # s without a traj_server command before holding position
NO_TRAJ_HOLD = 20.0 # s exploring with no new trajectory from EPIC: it is stuck (e.g. no flight corridor around the drone), go home
FINISH_HOLD = 10.0 # s EPIC must stay finished: it often reports no frontiers for a moment, then finds more and plans again


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def poly_samples(msg, step=0.2):
    """Positions along a PolyTraj: 8 coefficients per piece, highest power first."""
    points = []
    n = msg.order + 1
    for i, duration in enumerate(msg.duration):
        t = 0.0
        while t <= duration:
            point = []
            for coef in (msg.coef_x, msg.coef_y, msg.coef_z):
                value = 0.0
                for c in coef[i * n:(i + 1) * n]:
                    value = value * t + c
                point.append(value)
            points.append(tuple(point))
            t += step
    return points


class Mission:
    def __init__(self):
        self.home = rospy.get_param('~home', [6.0, 2.0, 0.2])
        self.takeoff_height = rospy.get_param('autonomous_flight/takeoff_height', 1.0)
        self.speed = rospy.get_param('autonomous_flight/desired_velocity', 1.0)
        self.yaw_rate = rospy.get_param('autonomous_flight/desired_angular_velocity', 1.0)
        waypoints = rospy.get_param('autonomous_flight/initial_waypoints', [])
        self.entry = [waypoints[i:i + 4] for i in range(0, len(waypoints), 4)]
        self.stall_seconds = rospy.get_param('~stall_seconds', 120.0)
        self.stall_gain = rospy.get_param('autonomous_flight/complete_min_gain', 0.25)

        self.state = 'IDLE'
        self.position = None
        self.setpoint = list(self.home)
        self.yaw = 0.0
        self.yaw_target = 0.0
        self.route = []
        self.fsm = ''
        self.traj_cmd = None
        self.traj_cmd_time = rospy.Time(0)
        self.crumbs = []
        self.best_block = 0.0
        self.best_block_time = None
        self.last_goal = None
        self.plan_time = 0.0
        self.triggered = False
        self.last_trigger = rospy.Time(0)
        self.finish_since = None
        self.explore_since = None
        self.last_traj_time = rospy.Time(0) # last new trajectory EPIC planned (traj_server repeats the last one forever)

        self.cmd_pub = rospy.Publisher('/planning/pos_cmd', PositionCommand, queue_size=10)
        self.state_pub = rospy.Publisher('/epic_mission/state', String, queue_size=10, latch=True)
        self.event_pub = rospy.Publisher('/epic_mission/event', String, queue_size=50)
        self.plan_pub = rospy.Publisher('/epic_mission/plan', String, queue_size=50)
        self.trigger_pub = rospy.Publisher('/waypoint_generator/waypoints', Path, queue_size=1)
        self.path_pub = rospy.Publisher('/dynamicExploration/bspline_trajectory', Path, queue_size=1)

        rospy.Subscriber('/quad_0/lidar_slam/odom', Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber('/epic_mission/traj_cmd', PositionCommand, self.traj_cmd_cb, queue_size=10)
        rospy.Subscriber('/planning/state', Marker, self.fsm_cb, queue_size=10)
        rospy.Subscriber('/planning/trajectory', PolyTraj, self.traj_cb, queue_size=10)
        rospy.Subscriber('/time_cost', Float32, self.time_cost_cb, queue_size=10)
        rospy.Subscriber('/exploration/status', String, self.status_cb, queue_size=1)

        rospy.Service('/exploration/start', Trigger, self.start_srv)
        rospy.Service('/exploration/return_home', Trigger, self.return_srv)
        rospy.Service('/exploration/land', Trigger, self.land_srv)
        rospy.Service('/exploration/pause', Trigger, self.no_pause_srv)
        rospy.Service('/exploration/resume', Trigger, self.no_pause_srv)

        self.set_state('IDLE')
        rospy.Timer(rospy.Duration(1.0 / RATE), self.tick)

    def set_state(self, state):
        self.state = state
        self.state_pub.publish(String(state))

    def event(self, text):
        rospy.loginfo('[epic_mission] ' + text)
        self.event_pub.publish(String(text))

    # ------------------------------------------------------------------ inputs
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.position = (p.x, p.y, p.z)
        if self.state not in ('IDLE', 'LANDED') and (not self.crumbs or dist(self.crumbs[-1], self.position) >= CRUMB_SPACING):
            self.crumbs.append(self.position)

    def traj_cmd_cb(self, msg):
        self.traj_cmd = msg
        self.traj_cmd_time = rospy.Time.now()

    def fsm_cb(self, msg):
        self.fsm = msg.text

    def time_cost_cb(self, msg):
        self.plan_time = msg.data / 1000.0 # EPIC publishes milliseconds

    def status_cb(self, msg):
        if self.state != 'EXPLORING':
            return
        block = json.loads(msg.data).get('explored_block_pct', 0.0)
        if self.best_block_time is None or block >= self.best_block + self.stall_gain:
            self.best_block = block
            self.best_block_time = rospy.Time.now()

    def traj_cb(self, msg):
        self.last_traj_time = rospy.Time.now()
        if self.state != 'EXPLORING' or self.position is None:
            return
        points = poly_samples(msg)
        if not points:
            return
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'map'
        for p in points:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)
        goal = points[-1]
        if self.last_goal is not None and dist(goal, self.last_goal) < GOAL_MOVED:
            return
        self.last_goal = goal
        length = sum(dist(a, b) for a, b in zip(points, points[1:]))
        self.plan_pub.publish(String(json.dumps({'goal': goal, 'from': self.position, 'path_length': length, 'legs': len(msg.duration),
                                                 'path_time': sum(msg.duration), 'plan_time': self.plan_time})))

    # ------------------------------------------------------------------ dashboard buttons
    def start_srv(self, _):
        if self.state != 'IDLE':
            return TriggerResponse(False, 'Mission already started (%s)' % self.state)
        if self.position is None:
            return TriggerResponse(False, 'No odometry yet')
        self.setpoint = list(self.position)
        self.crumbs = [self.position]
        self.route = [(self.setpoint[0], self.setpoint[1], self.takeoff_height)]
        self.set_state('TAKEOFF')
        self.event('Taking off')
        return TriggerResponse(True, 'Mission started')

    def return_srv(self, _):
        if self.state not in ('ENTRY', 'EXPLORING'):
            return TriggerResponse(False, 'Cannot return home now (%s)' % self.state)
        self.start_return('Return home requested')
        return TriggerResponse(True, 'Returning home')

    def land_srv(self, _):
        if self.state not in ('TAKEOFF', 'ENTRY', 'EXPLORING', 'RETURNING'):
            return TriggerResponse(False, 'Cannot land now (%s)' % self.state)
        self.setpoint = list(self.position)
        self.start_landing('Landing where it is: the ground below is not checked')
        return TriggerResponse(True, 'Landing')

    def no_pause_srv(self, _):
        return TriggerResponse(False, 'Pause is not available with EPIC')

    # ------------------------------------------------------------------ flight
    def start_return(self, reason):
        self.event(reason + ': retracing the flown path')
        self.route = self.home_route()
        self.setpoint = list(self.position)
        self.set_state('RETURNING')

    def start_landing(self, reason):
        self.event(reason)
        self.route = [(self.setpoint[0], self.setpoint[1], self.home[2])]
        self.set_state('LANDING')

    def home_route(self):
        """Shortest way home through remembered positions, so it only uses space the drone has already flown through."""
        crumbs = self.crumbs + [self.position]
        goal = (self.home[0], self.home[1], self.takeoff_height)
        goal_index = min(range(len(crumbs)), key=lambda i: dist(crumbs[i], goal))
        cells = {}
        for i, c in enumerate(crumbs):
            cells.setdefault(tuple(int(math.floor(v)) for v in c), []).append(i)

        def neighbours(i):
            cx, cy, cz = (int(math.floor(v)) for v in crumbs[i])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((cx + dx, cy + dy, cz + dz), ()):
                            d = dist(crumbs[i], crumbs[j])
                            # consecutive ones were flown directly, however far apart
                            if j != i and (d <= CRUMB_LINK or abs(i - j) == 1):
                                yield j, d

        start = len(crumbs) - 1
        best = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            cost, i = heapq.heappop(queue)
            if i == goal_index:
                break
            if cost > best[i]:
                continue
            for j, d in neighbours(i):
                if cost + d < best.get(j, float('inf')):
                    best[j] = cost + d
                    previous[j] = i
                    heapq.heappush(queue, (cost + d, j))
        route = [goal_index]
        while route[-1] in previous:
            route.append(previous[route[-1]])
        route.reverse()
        return [crumbs[i] for i in route] + [goal]

    def follow_route(self, speed, dt):
        """Moves the setpoint along the route. Returns its velocity, or None once the route is done."""
        if not self.route:
            return None
        left = speed * dt
        while self.route:
            d = dist(self.setpoint, self.route[0])
            if d > left:
                direction = [(q - p) / d for p, q in zip(self.setpoint, self.route[0])]
                self.setpoint = [p + left * v for p, v in zip(self.setpoint, direction)]
                return [speed * v for v in direction]
            self.setpoint = list(self.route.pop(0))
            left -= d
        return [0.0, 0.0, 0.0]

    def send(self, position, velocity, acceleration=(0.0, 0.0, 0.0), jerk=(0.0, 0.0, 0.0)):
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = 'world'
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.position.x, cmd.position.y, cmd.position.z = position
        cmd.velocity.x, cmd.velocity.y, cmd.velocity.z = velocity
        cmd.acceleration.x, cmd.acceleration.y, cmd.acceleration.z = acceleration
        cmd.jerk.x, cmd.jerk.y, cmd.jerk.z = jerk
        step = max(-self.yaw_rate / RATE, min(self.yaw_rate / RATE, wrap(self.yaw_target - self.yaw)))
        self.yaw = wrap(self.yaw + step)
        cmd.yaw = self.yaw
        cmd.yaw_dot = step * RATE
        self.cmd_pub.publish(cmd)

    def trigger_epic(self):
        now = rospy.Time.now()
        if self.fsm == 'WAIT_TRIGGER' and (now - self.last_trigger).to_sec() >= 1.0:
            self.last_trigger = now
            path = Path()
            path.header.stamp = now
            path.header.frame_id = 'world'
            pose = PoseStamped()
            pose.pose.position.z = 1.0 # EPIC ignores a trigger below -0.1: this is not a goal
            path.poses.append(pose)
            self.trigger_pub.publish(path)
        elif self.fsm and self.fsm not in ('INIT', 'WAIT_TRIGGER'):
            self.triggered = True

    def tick(self, _):
        dt = 1.0 / RATE
        velocity = [0.0, 0.0, 0.0]
        if self.state == 'TAKEOFF':
            velocity = self.follow_route(0.5, dt)
            if not self.route:
                self.route = [tuple(w[:3]) for w in self.entry]
                self.entry_yaws = [math.radians(w[3]) for w in self.entry]
                self.event('Flying %d entry waypoints' % len(self.entry))
                self.set_state('ENTRY')
        elif self.state == 'ENTRY':
            index = len(self.entry) - len(self.route)
            if self.route:
                self.yaw_target = self.entry_yaws[index]
            velocity = self.follow_route(self.speed, dt)
            if not self.route:
                self.event('Entry waypoints done: exploring')
                self.explore_since = rospy.Time.now()
                self.best_block_time = None
                self.set_state('EXPLORING')
        elif self.state == 'EXPLORING':
            if not self.triggered:
                self.trigger_epic()
            if self.fsm == 'FINISH':
                if self.finish_since is None:
                    self.finish_since = rospy.Time.now()
            else:
                self.finish_since = None
            # the last new trajectory, or the start of exploring if EPIC never planned one
            last_plan = max(self.last_traj_time, self.explore_since or rospy.Time(0))
            if self.fsm != 'FINISH' and (rospy.Time.now() - last_plan).to_sec() >= NO_TRAJ_HOLD:
                self.start_return('EPIC sent no trajectory for %.0f s (state %s)' % (NO_TRAJ_HOLD, self.fsm or 'unknown'))
            elif self.finish_since is not None and (rospy.Time.now() - self.finish_since).to_sec() >= FINISH_HOLD:
                self.start_return('EPIC finished: no frontiers left for %.0f s' % FINISH_HOLD)
            elif self.best_block_time is not None and (rospy.Time.now() - self.best_block_time).to_sec() > self.stall_seconds:
                self.start_return('Coverage gained under %.2f points in %.0f s' % (self.stall_gain, self.stall_seconds))
            elif self.traj_cmd is not None and (rospy.Time.now() - self.traj_cmd_time).to_sec() < CMD_TIMEOUT:
                c = self.traj_cmd
                self.setpoint = [c.position.x, c.position.y, c.position.z]
                self.yaw_target = c.yaw
                self.send(self.setpoint, (c.velocity.x, c.velocity.y, c.velocity.z), (c.acceleration.x, c.acceleration.y, c.acceleration.z),
                          (c.jerk.x, c.jerk.y, c.jerk.z))
                return
        elif self.state == 'RETURNING':
            if self.route and dist(self.setpoint, self.route[0]) > 0.1:
                self.yaw_target = math.atan2(self.route[0][1] - self.setpoint[1], self.route[0][0] - self.setpoint[0])
            velocity = self.follow_route(self.speed, dt)
            if not self.route:
                self.start_landing('Home: landing')
        elif self.state == 'LANDING':
            velocity = self.follow_route(0.3, dt)
            if not self.route:
                self.event('Landed')
                self.set_state('LANDED')
        self.send(self.setpoint, velocity or [0.0, 0.0, 0.0])


if __name__ == '__main__':
    rospy.init_node('epic_mission')
    Mission()
    rospy.spin()
