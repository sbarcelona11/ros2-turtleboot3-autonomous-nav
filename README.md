# TurtleBot3 autonomous DQN/Frontier exploration

ROS 2 Jazzy / Gazebo Harmonic exploration in ROBOTIS's official stage4 world.
The package name intentionally uses `turtleboot3_autonomous_nav` (two o's).
It builds an occupancy grid from LiDAR and odometry and selects six discrete
motion actions with a DQN. It does not launch Nav2, SLAM, AMCL, a map server,
teleoperation, or externally supplied goals.

The mission starts `coverage_mapper`, `observation_builder`, `dqn_explorer`, and
`safe_motion_controller`. Only the controller publishes `/cmd_vel`; its sensor
watchdog, clearance checks, and recovery logic can override policy actions.
Training replaces the explorer with `dqn_trainer`, including world/map resets,
replay updates, target-network synchronization, and epsilon-zero evaluations.

## How to run

There are **two** launchers. `mission.launch.py` explores the scenario with a
policy that already exists; `training.launch.py` trains a new one. Neither
starts Nav2, SLAM or AMCL: the map is built by `coverage_mapper` from ray
casting, and the pose comes from the simulator.

Everything runs **inside the container**. The LiDAR needs a rendering backend
even with no graphical interface, so `DISPLAY` has to point at the container's
X server:

```bash
export DISPLAY=:1
source /opt/ros/jazzy/setup.bash
source /home/ros/tb3_dependencies/install/setup.bash
source /ros2_ws/install/setup.bash
export TURTLEBOT3_MODEL=burger
```

### Mission: explore the scenario

```bash
ros2 launch turtleboot3_autonomous_nav mission.launch.py \
  explorer:=frontier use_rviz:=true use_gui:=true
```

This brings up Gazebo with the derived stage4 world, the mapper, the
observation builder, the safety controller, the selected policy and RViz. The
mission ends by itself for one of three reasons, which the explorer writes to
the log as `Mission ended: <reason>`: reaching `target_coverage`, exhausting
`max_steps`, or tipping over (`robot_tipped`). On termination it disables the
controller, which publishes zero velocity and rejects later actions until it is
explicitly enabled again.

| argument | default | what it does |
|---|---|---|
| `explorer` | `dqn` | `frontier` uses a deterministic breadth-first search; `dqn` uses the learned policy |
| `model_path` | empty | checkpoint path. **Required** with `explorer:=dqn`; ignored with `frontier` |
| `target_coverage` | `0.98` | fraction of `/coverage_reachable` at which the mission is considered finished |
| `max_steps` | `3000` | action cap, the safety net for a policy that gets stuck |
| `use_gui` | `true` | Gazebo interface |
| `use_rviz` | `true` | map, laser and coverage visualisation |
| `use_sim_time` | `true` | simulated clock in every node |

The learned policy needs its checkpoint given explicitly, so that a mission
cannot silently substitute untrained weights:

```bash
ros2 launch turtleboot3_autonomous_nav mission.launch.py \
  explorer:=dqn model_path:=/ros2_ws/models/best.pt
```

Load only trusted checkpoints: the current `.pt` format is Python pickle, not
TorchScript.

### Training: learn a policy

```bash
ros2 launch turtleboot3_autonomous_nav training.launch.py \
  episodes:=100 use_gui:=false model_directory:=/ros2_ws/models
```

This replaces the explorer with `dqn_trainer`, which also handles world and map
resets, replay buffer updates, target-network synchronisation and epsilon-zero
evaluations. At the episode limit it requests an acknowledged controller stop
and exits; stop the remaining simulator launcher with Ctrl-C.

| argument | default | what it does |
|---|---|---|
| `episodes` | `100` | training episodes. Evaluation episodes are additional |
| `model_directory` | `models` | where `best.pt` and `training_state.pt` are written |
| `training_config` | `config/training.yaml` | schedules, rewards and episode limits. Absolute path |
| `monitor` | `true` | prints metrics to the console while training |
| `use_gui` | `false` | Gazebo interface. See the warning below |
| `use_sim_time` | `true` | simulated clock in every node |

The default configuration evaluates every 10 episodes over 3 greedy runs, so
fewer than 10 episodes produces no checkpoint. Schedules and rewards are changed
with `training_config:=/absolute/path.yaml`.

#### Fresh aligned campaign and policy comparison

The observation now carries the `monitoring-map-v1` contract. Checkpoints and
training states made before that contract are intentionally rejected: start a
new campaign instead of resuming an unversioned model directory.

#### Historical performance reference

`models/best.pt` is the preserved `best_v7.pt` checkpoint. It is **not `monitoring-map-v1`** and is therefore a historical reference only: do not load it with the aligned DQN explorer or compare it as though it were a new aligned campaign.

| strategy | source | reachable coverage |
|---|---|---:|
| DQN v7 (legacy checkpoint) | episode 156, 3 greedy evaluations | 74.60 % |
| Frontier | current-scenario mission baseline | 97.97 % |

The DQN number is the best mean of three greedy evaluation episodes
(`best_v7.metrics.json`, 92,995 training steps); the Frontier number is the
documented one-run current-scenario mission baseline. They are useful as a
reference, but not a statistically fair DQN-versus-Frontier result. Use the
ten-run procedure below after training a `monitoring-map-v1` checkpoint to
make that comparison.

```bash
ros2 launch turtleboot3_autonomous_nav training.launch.py \
  episodes:=200 resume:=false \
  model_directory:=/ros2_ws/models/coverage-aligned-v1
```

After training, run the aligned policy greedily with its selected checkpoint:

```bash
ros2 launch turtleboot3_autonomous_nav mission.launch.py \
  explorer:=dqn model_path:=/ros2_ws/models/coverage-aligned-v1/best.pt
```

Compare it with Frontier over ten greedy mission runs per policy in the same
stage4 world. For each run, record the final `/coverage_reachable` value as the
mission coverage. A recorder started before each launcher captures both the
completion metric and the raw grid percentage for later inspection:

```bash
ros2 bag record /coverage_reachable /coverage_metrics
```

The explorer disables the controller at completion but both mission launchers
continue in `rclpy.spin()`. Use `Mission ended: <reason>` in the explorer log
as the end-of-run trigger: immediately stop the recorder with `Ctrl-C`, then
stop the mission launcher with `Ctrl-C`. The final `/coverage_reachable`
message already captured in that bag at the trigger is the mission coverage;
record that value before starting the next run. Then calculate the mean and
spread of the ten final values for DQN and Frontier. Mission launchers do not
publish `/training_episode_diagnostics`; that JSON topic is specific to
`training.launch.py`. Record it during the separate DQN training campaign if
you need safety interventions, turn actions and terminal reasons from its
greedy evaluation episodes, and inspect
`/ros2_ws/models/coverage-aligned-v1/best.metrics.json` with those training
diagnostics. Do not mix those training diagnostics with the ten-run mission
comparison.

For a one-episode integration exercise, which does **not** produce a useful
policy:

```bash
ros2 launch turtleboot3_autonomous_nav training.launch.py episodes:=1 \
  training_config:=/ros2_ws/src/turtleboot3_autonomous_nav/config/smoke_training.yaml \
  model_directory:=/ros2_ws/models/smoke
```

### Two warnings that cost time

**Do not launch twice without cleaning up.** Two Gazebo servers publish the same
`/clock` and the same TF, and the symptom is not an error but RViz complaining
about extrapolation and a map that rotates. Check that none is left alive before
relaunching. If you kill processes with `pkill -f`, do it from a **script**: the
pattern matches the command line of the very shell running it, so it kills
itself halfway through the cleanup.

**The LiDAR needs rendering.** With `use_gui:=false` and no valid `DISPLAY`, the
laser drops from 5.3 Hz to 0.12 Hz and training crawls without reporting any
error at all.

## Relation to the PRIA assignment

The mission starts from the TurtleBot3 Burger's initial pose in the
`turtlebot3_dqn_stage4` scenario. Once `mission.launch.py` is running, the robot
monitors as much observable area as it can with no manual driving commands. The
architecture separates information gathering, decision and actuation
explicitly:

```text
/scan + /odom
      │
      ├─ coverage_mapper ──> /coverage_map + /coverage_metrics
      ├─ observation_builder ──> /dqn_observation
      ├─ dqn_explorer ──> /exploration_action
      └─ safe_motion_controller ──> /cmd_vel
```

During training, `dqn_trainer` replaces `dqn_explorer` and `training_monitor`
watches `/training_metrics` and `/coverage_map` without publishing.

| Assignment criterion | Implementation and evidence |
| --- | --- |
| ROS 2 compatible environment | ROS 2 Jazzy, Gazebo Harmonic and the official `turtlebot3_dqn_stage4.launch.py` launcher. |
| Nodes written by the student | `coverage_mapper`, `observation_builder`, `safe_motion_controller`, `frontier_explorer`, `dqn_explorer`, `dqn_trainer`, `simulation_localizer`, `exploration_visualizer` and `training_monitor`. |
| Effective perception | `/scan` identifies free and occupied space; `/odom` places the rays on a grid. Together they form `/coverage_map` and the 90-value DQN observation, which includes the scenario map and the robot's pose within it. |
| Autonomous decision | The DQN policy picks among five actions from sectored LiDAR, the scenario map, the pose and the directional unknown-area gains. It is not a timed sequence: without fresh sensor data the controller stops the robot. |
| ROS 2 actuation | Only `safe_motion_controller` publishes `geometry_msgs/Twist` on `/cmd_vel`; it bounds speed, brakes on risk and recovers from being stuck. Verifiable with `ros2 topic info /cmd_vel`. |
| No teleoperation | `mission.launch.py` loads a mandatory checkpoint and starts no teleoperation, navigation, SLAM or localisation node, and waits for no external goals. |
| Monitored area | `/coverage_reachable` publishes the monitored percentage of the reachable region, and `best.metrics.json` stores the mean coverage of the greedy evaluations. |
| Safety and stability | LiDAR, odometry and action watchdogs, a stop on invalid or stale data, bounded recovery, step and coverage limits, and a final stop at zero velocity. |

A cell counts as monitored once it has been observed from closer than
`inspection_radius` metres by a LiDAR measurement projected through odometry.
The mission percentage is the monitored fraction of the reachable region inside
the grid, which covers exactly the 5 m × 5 m room of the stage4 scenario (walls
at ±2.425 m). It includes occupied cells and is not the same as reachable free
area.

The grid has to match the scenario. A larger grid leaves permanently unknown
cells outside the walls: it sinks the reported percentage (a 25 m² room inside a
400 m² grid saturates at about 6 %, putting any high target out of reach) and
biases the exploration gains towards the outside of the scenario, where the
robot can never go.

### Monitoring means getting close, not merely seeing

A cell counts as monitored only if it was observed from closer than
`inspection_radius` metres. The occupancy map is still built at the LiDAR's full
range, because navigation, the policy observation and the reachable-region
computation all need it; proximity is required by the mission metric alone.

Without that requirement the mission cannot be learned. A 3.5 m LiDAR resolves
almost all of a 4.85 m room from a single sweep, so coverage saturates without
the robot moving and a random policy scores nearly as well as a good one: there
is no gap left to optimise. With `inspection_radius: 1.5`, the roughly 22.4 m²
of free area force a 3 m wide corridor to be swept, about 7.5 m of travel.
Measured in the scenario: a random policy reaches **52 %** in 500 steps, while
one that advances steadily reaches 100 %.

The radius sets the difficulty. A smaller one widens the gap but requires
raising `max_steps`, because not even a perfect policy finishes the route in
time.

### Frontier exploration

`mission.launch.py` accepts `explorer:=frontier`, which replaces the learned
policy with a breadth-first search to the nearest reachable cell that has not
been monitored yet. It publishes on the same `/exploration_action`, so the
safety controller and the rest of the mission are unchanged, and `/cmd_vel`
still has exactly one publisher.

It exists because the learned policy harvests the easy area and then circles
ground it has already monitored: crossing known space earns nothing. The search
has no such blind spot and reached **99.1 %** coverage against 72.5 % for the
trained DQN.

Those two figures were measured with the cylinders still moving and with the
100×100 grid from −2.5 m, so they are not comparable with anything measured
afterwards. The comparison was redone on the current scenario, with a policy
retrained from scratch over 100 episodes against the current dynamics:

| | coverage |
|---|---|
| frontier explorer, mission | **97.97 %** |
| DQN, best evaluation during training (episode 78) | 71.44 % |
| DQN, mission with that checkpoint | 51.02 %, on exhausting 3000 steps |

The learning itself worked: evaluation rose from 60.28 % at episode 13 to
71.44 % at episode 78. Almost all of it happened early, though. Between
episodes 26 and 78, while epsilon fell from 0.55 to 0.16 and the learned policy
took over from random exploration, the agent gained 1.3 points, and the final
evaluation at minimum epsilon came in below the record at 55.1 % and 57.7 %.

The 20-point gap between the 71.44 % evaluation and the 51.02 % mission is the
same checkpoint measured two ways. Evaluation averages three episodes of 120
simulated seconds; the mission runs to 3000 steps with no time limit. Watching
the mission says why that matters: the robot sat motionless 18 cm from the east
wall, at (2.249, −0.659), choosing `SOFT_RIGHT` on every single decision. It
pushes into the wall, the safety controller blocks the translation and turns it,
and the policy immediately asks for the same thing again. A short episode ends
before the policy can jam; a long mission gives it all the time it needs to. It
also lands in the 0.15–0.20 m band, too far for the contact escape to fire and
close enough that the front stays blocked.

The DQN learned to move and to harvest the easy half of the room. It did not
learn to explore.

#### Rewarding the journey, not only the arrival

Four changes followed from that diagnosis, and they moved the mission result
from 51.0 % to 87.5 %:

1. **Travel counts as progress.** `stall_seconds` ended an episode after 25
   simulated seconds without a new cell, and crossing this arena at 0.10 m/s
   takes about 48, so an agent that correctly set off for the unexplored far
   side was cut short every time - 117 of 127 episodes ended on `stalled`. A
   step now counts if it discovers cells **or** closes ground towards the
   nearest frontier, measured with the same breadth-first search the
   deterministic explorer uses.
2. **A reward for approaching.** `frontier_approach_reward` pays per cell of
   ground closed. It is the plain distance difference, not the gamma-discounted
   potential of Ng, Harada and Russell: with gamma below one that formulation
   pays for standing perfectly still, which is the behaviour being corrected.
3. **`intervention_penalty` from 1.00 to 0.10.** At the measured decision rate
   one intervention cost as much as twenty-three steps of doing nothing, so
   avoidance dominated the return and the policy stopped approaching anything.
4. **Recovery can no longer be disarmed by turning.** The controller reset its
   progress clock on any action that did not translate, which a policy could
   satisfy by spinning, keeping automatic recovery asleep forever.

The first change needed a second pass, and the failure is worth recording. Its
first version accepted any decrease in frontier distance, and that distance
drifts by a cell on its own as the laser resolves the map, so a spinning robot
kept resetting the stall clock. Turning in place had been punished by ending
the episode and forfeiting the rest of its return; that pressure disappeared.
Stall terminations fell to 1 in 39 and the policy trained under it spun through
79.8 % of its decisions - worse than the 63.4 % that started all this.
Requiring `progress_displacement` metres of real ground covered fixed it: a
turn in place covers none, whatever the map does around it.

| campaign | episodes | transitions | evaluation | mission coverage |
|---|---|---|---|---|
| original reward | 100 | 30 144 | 71.44 % | 51.02 % |
| shaped, first version | 30 | 7 798 | 60.68 % | 53.12 % |
| shaped | 13 (simulator crashed) | - | 68.50 % | 60.40 % |
| shaped | 30 | 15 204 | 62.11 % | 87.48 % |
| shaped, paced simulation | 170 | **92 995** | **74.60 %** | **97.59 %** |

#### Experience was the binding constraint all along

Every campaign up to the last one collapsed onto a single action. With the
original reward the policy turned in place on 63.4 % of its decisions; with the
shaped reward at 30 episodes it drove forward on 100 % of them, and reached
87.5 % only because the safety controller supplied the steering. Both are the
same failure wearing different clothes, and reward weights were never going to
fix it: 15 204 transitions is roughly a tenth of what this algorithm needs.

Reaching 92 995 transitions took the pacing work described under **Simulation
pacing** below. At 170 episodes the policy stops collapsing and starts
steering. Measured over two minutes of mission:

| action the policy asks for | 100 ep, original | 30 ep, shaped | 170 ep, shaped |
|---|---|---|---|
| forward turning gently left | 26.1 % | 0 % | 49.0 % |
| forward | 10.2 % | 100 % | 24.0 % |
| turn in place right | 40.7 % | 0 % | 17.0 % |
| forward turning gently right | 0.3 % | 0 % | 7.9 % |
| turn in place left | 22.7 % | 0 % | 2.1 % |
| safety interventions | 50 | 2 | 1 |

All five actions are in use, three quarters of the decisions ask for
translation with a heading correction, and turning in place is kept for the
turns that need it.

The learned policy ends up 0.4 points short of the deterministic explorer,
which is closer than this comparison deserves to be read as a tie. The
explorer reached 97.97 % in **518 decisions** and stopped because it had
finished; the policy reached 97.59 % having spent its whole **3000 step**
budget. Same ground covered, a route nearly six times longer. An exact
breadth-first search over the full map is still the better way to clear a room
this size; what the learner now demonstrates is that the behaviour is
reachable from a 90-value observation and five discrete actions.

Three details make it work, each one found by measuring:

- **It plans on the monitoring map**, not the occupancy map. A cell resolved by
  a distant ray is no longer unknown but is still unmonitored: searching on
  occupancy, the mission declared itself complete at 81.4 %.
- **It inflates obstacles** by the controller's braking distance
  (`clearance_cells_for`). A cell-level path grazes the walls and safety braked
  on nearly every step: 133 interventions in thirty seconds.
- **It aims several cells ahead** (`lookahead_cells`). Aiming at the adjacent
  cell made the heading swing and the robot pivoted instead of driving: 87 of
  141 actions were turns in place.

### Localisation in simulation

Wheel odometry is unusable here. Measured against Gazebo's true pose in the
middle of a mission: **163.8° of heading error and 1.3 m of position**, in a
4.85 m room. The robot slips on every contact and nothing closes the loop, and
since every ray is placed with that pose, the map bore no relation to the
scenario.

The derived world includes Gazebo's `OdometryPublisher` system on the Burger,
which derives the pose from the model's true state and publishes it on a topic
of its own, `/odom_truth`. A per-model topic is what makes this usable: bridging
the list of every object's pose loses the names, and nothing can then tell the
robot from an obstacle.

`simulation_localizer` republishes that odometry as the `odom → base_footprint`
transform, and the one Gazebo publishes from the wheels is diverted to
`/tf_wheel`. Without this, RViz drew the robot and its laser a metre away from
the correct map. Verified: **0.00 m** between the transform and the true pose,
and the map matches the walls to **8 cm** with all seven inner walls in place.

**This is information a real robot does not have**: a physical deployment would
need SLAM, which the assignment excludes. What remains sensor-driven is what is
being assessed: the occupancy map, the frontier search and the safety controller
all work from LiDAR.

### Contacts and tipping

The controller slows down as it approaches an obstacle, ahead (`slow_distance`)
and to the side (`side_slow_distance`), and recovery **reverses** when the rear
sector is clear. Turning does not free a robot whose nose is against a wall:
with no reverse it stayed pinned for ninety seconds through 147 interventions.

Lateral braking exists because the two moving cylinders reach the robot from the
side, where forward braking does nothing. With both, and a nominal speed of
0.10 m/s, a mission went from tipping at step 73 with 58.6 % to lasting 1624
steps and reaching **96.9 %**.

Even so the robot ends up tipping over, and the problem is **still open**. When
it happens the controller detects it from the IMU, stops the motor and publishes
it on `/robot_tipped`; the mission ends with that reason instead of continuing
to issue commands to a robot lying on its side.

#### What was ruled out, and by which measurement

Four hypotheses were tried and fell. They are written down because each looked
reasonable and measurement was the only thing that separated them.

1. **Bounding the physics contact correction.** The world declares
   `<physics type="ode">` and a `contact_max_correcting_vel` of 2000 m/s.
   Lowering it to 100 and then to 1 changed nothing, and the reason is that
   **nothing reads it**: the server log says
   `Loaded [gz::physics::dartsim::Plugin]`. The engine is DART, and the whole
   `<ode>` block is Gazebo Classic configuration that gz-sim ignores. The
   setting and its test were removed; a test that checks a value the simulator
   never consults only gives false assurance.
2. **The teleporting obstacles.** `Obstacle1Plugin` and `Obstacle2Plugin` move
   their cylinder with `SetWorldPoseCmd` from a wall clock, so they jump without
   travelling through the space between and can appear inside the robot. It is a
   real defect and the plugins are stripped from the derived world, but the
   tipping continued without them.
3. **A laser blind at close range.** The composition of the scan was measured
   with the robot resting against an obstacle: it clips cleanly at its 0.12 m
   minimum, with no zeros and no infinities in the front sector. The controller
   **does** see the obstacle.
4. **Turning in place on contact.** The forensics showed `cmd=(0.000, -1.000)`
   with the height rising from 0.023 to 0.029 m: the robot climbs while turning.
   Reversing already existed but was gated behind `stalled`, which takes twenty
   seconds, and the climb happens in under three. Front contact now triggers the
   reverse by itself (`contact_distance`, see `test_contact_escape.py`). It
   improved the behaviour, but the tipping remained.

#### The live hypothesis

Upstream's Burger model declares the wheels with `mu = mu2 = 100000.0` and
`slip1 = slip2 = 0`, and the rear caster as a 5 mm sphere with no friction
declared. Effectively infinite friction with no slip is a Gazebo Classic idiom
for ODE; under dartsim it means the wheel **can never slip**, so when the robot
touches something the only resolution left to the solver is to rotate the body.
It matches the measured signature: a pure 48.7° pitch with zero roll, and peaks
of 174° in other runs. Still to be confirmed with a derived copy of the model
carrying realistic friction.

### Two coverage metrics

`/coverage_metrics` publishes the fraction over the whole grid. It is stable, so
the discovery reward is derived from it.

`/coverage_reachable` publishes the observed fraction **of the region the robot
can still reach**, and it is what defines mission end, evaluation and
checkpoints. The region grows from observed free space through every cell that
is not occupied: the inside of a wall or an obstacle is sealed off by occupied
cells and excluded, while the shadow behind an obstacle is connected and counts
as pending work. A finished exploration therefore reads as about 100 % instead
of being capped by unobservable cells.

#### The grid has to stop at the wall

That exclusion only works while the observed wall face is a continuous line of
occupied cells. It never is: a distant wall is sampled by fewer beams and a cell
needs two hits to read as occupied, so the face comes out perforated and the
reachable region **escapes through the holes**.

The grid ran from −2.5 to +2.5 m with the walls at ±2.425, and that 0.075 m rim
is wall body and outside ground. Measured on a finished run: of 680 cells
counted as pending, **456 were outside the room**. Two thirds of the shortfall
was ground no laser can reach. Sizing the grid to the room exactly, 97×97 cells
of 0.05 m from −2.425, removes the leak at its source, with no sealing logic
that could hide a real gap.

| | 100×100 grid from −2.5 | 97×97 grid from −2.425 |
|---|---|---|
| mission coverage | 91.2 % | **97.97 %** |
| pending cells outside the room | 456 | 0 |
| pending cells inside | 224 | 87 |
| steps to finish | 3000 exhausted | 518, on reaching the target |

The remaining 87 cells are about 0.22 m² of floor in the strip against the
walls, where the robot cannot put its centre because of the 0.20 m braking
margin. Both numbers measure the same work by the robot; what changes is what
counts as discoverable.

The stage4 scenario is not an empty room: it has seven 1 m inner walls forming a
maze, plus two cylindrical obstacles. Accumulated-evidence mapping tolerates a
cell changing state, because it returns to free once the free-space evidence
outweighs the occupied evidence.

### Simulation pacing

The world asked for a real-time factor of one, and the campaign obeyed it: the
achieved factor measured 0.97 to 0.99 over a full run. Nothing was short of
machine. Total CPU use was about 190 % of the 1200 % available, with the Gazebo
server at 65 %, the mapper at 43 %, the controller at 34 %, the observation
builder at 29 % and the bridge at 18 %. The simulation was pinned to the wall
clock, not to the hardware.

Raising the request to 3.0 in the derived world achieves **2.75**. The
campaign that needed six hours now takes two and a half, which is what made
92 995 transitions practical in one sitting.

This is safe for the experiment because every budget in this package is
denominated in simulated time: episode length, the stall budget, the penalties
charged per simulated second, and the decision throttle. Verified rather than
assumed - at the higher pace the decision rate held at **4.94 per simulated
second** against 4.95 before, and episodes still reached 578 to 595 steps. The
pipeline keeps up with nearly triple the messages per wall second, so the agent
sees exactly the problem it saw before. The physics step is untouched at
0.001 s, so the solver sees what it saw too.

Three things were investigated first and were not the cause, each discarded by
measurement rather than argument: the episode length (episodes already reached
594 decisions against the 518 the deterministic explorer needs), the IMU at
172 Hz (`/clock` publishes at 965 Hz, five times more), and the reset between
episodes (0.6 s, against the 120 s the episode itself runs).

### Documented deviation: the cylinders do not move

The two cylindrical obstacles of the official stage4 are moved by
`Obstacle1Plugin` and `Obstacle2Plugin`, which **teleport** them: they call
`SetWorldPoseCmd` on every update with a pose computed from
`std::chrono::steady_clock`, that is wall time and not simulated time. When the
simulation runs slower than real time, the cylinder covers the elapsed seconds
in a single physics step; it does not travel through the space between, it
appears at the far end. Landing inside the robot is a deep interpenetration that
the solver resolves with an impulse. Measured with the simulator's own pose: the
robot reached 8.1 m of altitude in one sample and 16.9 m in the next, fourteen
metres outside a 4.85 m room.

The plugins are stripped from the derived copy of the world; the cylinders stay
where they are, with their geometry and the shadows they cast, which is what
makes the exploration worth doing. The upstream world is untouched. The
consequence is that the scenario is somewhat **easier** than the official
stage4, and it is declared here as a deliberate deviation in the face of an
upstream defect, not as a simplification of the problem.

### Episodes invariant to simulation rate

The episode budget is **simulated time** (`max_episode_seconds`), not a step
count, and `stall_seconds` measures stalling the same way. A step is one
observation, so counting steps tied the episode to the publication rate, which
varies with machine load and Gazebo's real-time factor: measured in this
project, the robot advanced 12 mm per step in one run and 2.4 mm in the next.
The same `max_steps` bought five times less travel, coverage figures stopped
being comparable between runs, and the value function was trained against a
horizon shifting underneath it. `max_steps` remains only as a guard in case the
simulated clock stops advancing.

For the same reason, `step_penalty` and `no_progress_penalty` are charged **per
simulated second**. Discovering a cell, reaching a frontier or triggering the
safety controller are events and are charged once.

`max_observation_rate` sets the decision rate at 10 Hz. The simulated LiDAR
publishes above 40 Hz, but a robot at 0.15 m/s advances less than four
millimetres between those sweeps: deciding at that rate fills the replay memory
with nearly identical transitions and multiplies the computation per episode
without adding information.

### Mapper load

Tracing the 360 rays of a sweep costs tens of milliseconds, and sweeps arrive
much faster than the map needs updating. Without a limit the scan callback
leaves no room for the reset service on a single-threaded executor, the episode
reset times out and training aborts. `max_scan_rate` bounds the integration
rate; at 8 Hz the robot moves less than two centimetres between accepted sweeps,
so the map loses no quality.

For the same reason `reset_timeout_seconds` is 30 s: the Gazebo server and GUI
saturate the machine while the world loads, and a tight deadline aborts training
through contention rather than through a real hang.

An exhausted reset is also retried up to `reset_retry_limit` times before
aborting. A reset is a request to the simulator between episodes, with the robot
already stopped; treating a slow response as fatal cost 18 training episodes in
one campaign. Sensor data timeouts do abort immediately: moving a robot on stale
readings is unsafe.

### Action space

The policy chooses among five actions: drive forward, drive forward turning
gently to each side, and turn in place to each side. `RECOVER` is **outside**
the policy's space: it holds the robot turning for `recovery_duration` seconds,
while the other actions last until the next one. Exposed to the policy, a single
choice dominates the time budget; an untrained agent picking it one time in six
spends most of the episode motionless.

For the same reason the stall clock only runs while the commanded action asks
for translation. Turning in place never translates the robot, and since recovery
also turns, counting it as lack of progress made recovery renew the very
condition that triggered it.

### Scenario-scale observation

The policy can only head for unexplored area it perceives. The 90-value vector
published on `/dqn_observation` describes the whole scenario, not just the
immediate surroundings:

| indices | contents |
| --- | --- |
| `[0:12]` | per-sector LiDAR minima (nearby obstacles) |
| `[12:20]` | unknown fraction of each 45° sector measured over the **whole** map |
| `[20:84]` | scenario map reduced to 8×8 in a fixed frame (`-1`/`0`/`1`) |
| `[84:86]` | robot position normalised to `[-1, 1]` per axis |
| `[86:88]` | heading as sine and cosine |
| `[88:90]` | linear and angular velocity |

The directional gains are measured over the entire map: unexplored area metres
away still produces a signal. The reduced map in a fixed frame, together with
the normalised position, lets the policy tell which areas it has already
covered. Fine nearby detail comes from the LiDAR sectors.

A block of the reduced map is marked occupied if it holds any occupied cell, so
obstacles survive the reduction, and free if it holds any resolved cell; only
blocks that are entirely unvisited stay unknown.

### Resuming after a simulator crash

The trainer saves its progress to `model_directory/training_state.pt` at the end
of every episode and picks it up on the next launch; `resume:=false` forces a
fresh campaign.

It is needed because the simulator crashes. In one campaign of this project,
Gazebo aborted at episode 27 with `ODE INTERNAL ERROR 1: assertion "aabbBound >=
dMinIntExact && aabbBound < dMaxIntExact" failed in collide()`, inside DART's
collision detector during a world reset: a body reached an overflowed bounding
box. It is an upstream defect, outside this package's control. When the server
dies `/clock` stops, so observations stop being published and the trainer times
out. What is within our control is not losing the work: a crash costs one
episode, not the campaign.

The replay memory is **not** saved, being large and already excluded by the
checkpoint format, so a resume continues from the learned weights and refills
the buffer.

### Learning stability

The update uses **Huber loss** (`huber_delta`), gradient norm clipping
(`gradient_clip`) and a bounded per-step reward (`reward_clip`).

These are not decorations. With a quadratic loss the gradient grows without
bound with the temporal error, and a single surprising transition can ruin the
network. In one campaign of this project the policy degraded monotonically
across five evaluations — 59.9 %, 57.2 %, 53.5 %, 43.7 %, 35.7 % — while epsilon
fell from 0.97 to 0.23 and the learned policy took over from random exploration.
Episodes ending stuck went from 6 of 16 to 11 of 16 in the same period: the
signature of a diverging Q function. A random policy reached 52 %, so the
trained agent had fallen below chance.

A non-positive value disables each clip, so the effect can be measured.

### Exploration during training

`epsilon_decay` is not configured: it is derived from `epsilon_start`,
`epsilon_min` and `max_episodes`, so exploration finishes annealing exactly as
the configured episodes complete. A fixed constant drifts out of step with the
run length; 0.995, for example, needs about 600 episodes to fall to a floor of
0.05, so a 100-episode run would end with the agent still taking more than 60 %
of its actions at random and showing no visible progress.

### Console monitor

`training.launch.py` starts `training_monitor`, which publishes nothing and only
reads `/training_metrics` and `/coverage_map`. It prints episode, step, scenario
coverage, grid coverage, accumulated reward, epsilon, best mean evaluation
coverage and steps per second. It is disabled with `monitor:=false` and can also
be run separately during a campaign:

```bash
ros2 run turtleboot3_autonomous_nav training_monitor
```

In an interactive terminal it redraws a single line; under the launcher, where
every write appears as a prefixed line, it emits one update every
`update_interval` seconds.

### Demonstration protocol

1. Train, or select a checkpoint produced by the training launcher.
2. Launch `mission.launch.py` alone with that checkpoint; do not start `teleop`,
   and do not publish on `/cmd_vel` by hand.
3. Show `/coverage_map`, `/scan` and `/exploration_status` in RViz while the
   robot navigates.
4. Verify `ros2 topic info /cmd_vel --verbose`: the only publisher must be
   `/safe_motion_controller`.
5. On finishing through target coverage or `max_steps`, record
   `/coverage_metrics`, the final grid and the `best.metrics.json` file.

The reference video defines the mission's visual specification. Before the final
demonstration, the initial pose, the time available, the visible events and the
expected coverage percentage must be checked against it. Those values are
adjusted without changing the architecture through `target_coverage`,
`max_steps` and the configuration files.

## Container and source dependencies

From VS Code, use **Dev Containers: Rebuild and Reopen in Container** after
changing `.devcontainer/Dockerfile`. The container needs ROS 2 Jazzy, Gazebo
Harmonic, `ros_gz`, RViz, NumPy, and pytest. The network must be available on the
first launch so Gazebo can cache the upstream ground-plane model.

Install the official simulation sources in a separate dependency workspace
inside the container. The package does not need the upstream DQN training stack.
These commands assume that `/ros2_ws/src/turtleboot3_autonomous_nav` is mounted:

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p /home/ros/tb3_dependencies/src
git clone --branch jazzy https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git \
  /home/ros/tb3_dependencies/src/turtlebot3_simulations
git -C /home/ros/tb3_dependencies/src/turtlebot3_simulations checkout \
  45633014a14e8f438495b532a723e4ad45cbbd31
sudo apt-get update
sudo apt-get install -y ros-jazzy-ros-gz ros-jazzy-rviz2 python3-numpy python3-pytest
cd /home/ros/tb3_dependencies
colcon build --packages-select turtlebot3_gazebo
source install/setup.bash
cd /ros2_ws
colcon build --packages-select turtleboot3_autonomous_nav
source install/setup.bash
pytest src/turtleboot3_autonomous_nav/test -v
```

Source the dependency workspace and `/ros2_ws/install/setup.bash` in each new
terminal. The launchers select Burger and include the official
`turtlebot3_dqn_stage4.launch.py`. A narrow adapter gates upstream's GUI-only
include, and the launchers add the official obstacle plugin library directory.
The official stamped-velocity subscription is remapped; a one-way bridge
connects this package's `geometry_msgs/Twist` to Gazebo's velocity topic.

## Runtime and safety protocol

The commands and every launch argument are in **How to run** above; this section
covers what the nodes do once they are running. Both launchers default to
`use_sim_time:=true`, and the GPU LiDAR needs a working rendering backend even
when the GUI is disabled.

A mission stops at `max_steps` actions, at `target_coverage`, or on
`robot_tipped`, whichever comes first. `model_path` is required with the DQN
explorer so a mission cannot silently substitute untrained weights, and the
explorer also rejects an absent or empty model when run directly with `ros2 run`.
On termination it disables the controller, which publishes zero velocity and
rejects later actions until explicitly enabled for a new run.

The controller starts disabled. The policy nodes use the acknowledged
`/safe_motion_controller/enable` (`std_srvs/SetBool`) service to enable motion.
Both disabling and enabling clear cached action, scan, odometry, coverage, and
recovery state, publish zero velocity, and establish a DDS publication cutoff.
Motion requires new scan, odometry, and action publications after that cutoff.
Scan and odometry expire after 0.5 seconds, actions after 1 second, using both
simulated age and a wall-clock watchdog. Negative ages after clock rewind also
stop motion. Automatic stall recovery lasts at most `recovery_duration` (2 seconds
by default), then gives forward policy actions another progress interval.

Each training reset acknowledges controller disable before resetting Gazebo,
then clears the mapper and `/observation_builder/reset` (`std_srvs/Trigger`).
The builder clears all cached inputs, rejects earlier DDS publications, and tags
observations with its new episode epoch. Only after these acknowledgements does
the trainer enable the controller and wait for fresh sensors and a matching
observation. Service discovery, responses, fresh sensors, and subsequent
observations each have a wall-clock `reset_timeout_seconds` deadline (10 seconds
by default). A timeout aborts training and requests a controller stop.

## Observe autonomous operation

RViz uses `odom` as its fixed frame and displays `/coverage_map`, `/scan`, and
`/odom`. A read-only visualizer converts `/cmd_vel` and `/coverage_metrics` into
standard RViz markers on `/exploration_status`: the command arrow and text show
linear/angular commands and current coverage. No custom RViz plugins are needed.

To demonstrate autonomous operation, start only the mission, leave all keyboard
control nodes stopped, and watch the map reveal new cells as the robot moves.
In a second terminal:

```bash
ros2 topic echo /exploration_action
ros2 topic echo /cmd_vel
ros2 topic echo /coverage_metrics
ros2 topic info /cmd_vel --verbose
ros2 node info /safe_motion_controller
ros2 node list
```

Stop each `topic echo` with Ctrl-C before the next command. The velocity topic
must have one publisher named `/safe_motion_controller`. The graph must contain
no navigation, localization, map-server, or teleoperation nodes. Coverage is the
fraction of observed cells in the arena-sized grid, including occupied cells; it
is not the fraction of reachable free space. Cells inside obstacles and behind
walls never resolve, so step and stall limits also end episodes. Short smoke
results establish integration, not policy quality.

For a bounded automated check while the mission is running:

```bash
python3 /ros2_ws/src/turtleboot3_autonomous_nav/test/integration_probe.py --seconds 30
```

It reports topic counts, first/last known-cell counts, velocity publishers, and
node names, and exits unsuccessfully if coverage does not grow, motion is absent,
or another velocity publisher or excluded node is present.

## Outputs and limits

`model_directory` receives `best.pt` and `best.metrics.json` whenever mean
evaluation coverage improves. The checkpoint preserves online/target weights,
network dimensions, configuration, and evaluation metrics. It does not contain
replay memory or optimizer state for exact training resumption. JSON includes
mean coverage, total episode index, evaluation count, and training step count.

Live topics include `/coverage_map` (OccupancyGrid), `/coverage_metrics`
(Float32), `/dqn_observation` (86 floats), `/exploration_action` (Int32),
`/safety_intervention` and `/recovery_active` (Bool). `/training_metrics` contains
episode, step, coverage, cumulative reward, epsilon, and best mean evaluation
coverage, in that order. Maps are live only. Record the
topics with `ros2 bag record` if persistent demonstration data is needed.

Mapping uses odometry without loop closure, and the policy is environment
specific. Run a full training/evaluation campaign before making claims about
coverage or collision performance.

Training loads Burger into Gazebo's initial world state. Gazebo Sim 8.11.0
removes a Burger spawned after startup when `reset.all` restores that initial
state, which stops odometry even though an old laser sensor can keep publishing.
The training launcher derives a temporary copy of the official stage4 world
with the standard Burger model included at the origin and removes it on launch
shutdown. The upstream world, moving obstacles, robot state publisher, and
bridges are preserved; the later robot creation process is omitted.

Full world reset, controller disable/enable acknowledgements, mapper and
observation reset epochs, and fresh sensor requirements remain enforced. A
Gazebo 8.11.0 smoke run completed two training and two evaluation episodes and
saved a reloadable checkpoint. See the
[reset investigation and verification](../../../docs/verification/2026-09-05-gazebo-training-reset.md)
for the reproduction and partial-reset comparisons. This verifies the episode
integration, not a learned policy's exploration quality.
