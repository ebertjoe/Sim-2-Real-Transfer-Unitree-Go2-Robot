# deploy_rl_policy

ROS2 package for running a trained RL locomotion policy on the Unitree Go2 — either in a MuJoCo simulation or on the real robot.

The stack consists of four nodes:

| Node | Type | Purpose |
| --- | --- | --- |
| `mujoco_simulator.py` | Python | MuJoCo physics simulation of the Go2 (simulation only) |
| `low_level_ctrl` | C++ | Finite state machine (laying down → standing up → executing policy) and joint-level PD control |
| `rl_policy.py` | Python | Runs policy inference and publishes joint targets on `/rl/target_pos` |
| `joy_node` | external | Reads a gamepad and publishes `sensor_msgs/Joy` |

---

## Quick start (simulation)

Build and source the workspace first:

```bash
colcon build --symlink-install
source install/setup.bash
```

Then start the nodes **in this order**, each in its own terminal.

### 1. MuJoCo simulator

```bash
ros2 run deploy_rl_policy mujoco_simulator.py
```

### 2. Low-level controller

```bash
ros2 run deploy_rl_policy low_level_ctrl --ros-args \
  -p is_simulation:=true \
  -p policy_kp:=25.0 \
  -p policy_kd:=0.5
```

### 3. RL policy node

```bash
ros2 run deploy_rl_policy rl_policy.py --is_simulation True
```

### 4. Joystick node

```bash
ros2 run joy joy_node
```

---

## Parameters

| Parameter | Node | Value | Description |
| --- | --- | --- | --- |
| `is_simulation` | `low_level_ctrl` | `true` / `false` | Selects the simulation or the real-robot interface |
| `policy_kp` | `low_level_ctrl` | `25.0` | Proportional gain used while the policy is active |
| `policy_kd` | `low_level_ctrl` | `0.5` | Derivative gain used while the policy is active |
| `--is_simulation` | `rl_policy.py` | `True` / `False` | Same switch on the policy side |

> The PD gains must match the ones the policy was trained with. Launching `low_level_ctrl` without the overrides falls back to the defaults, which will not match the trained policy.

---

## Sending joy commands without a gamepad

If no physical controller is connected, the same commands can be published manually on `/joy`.

### Stand up

Published once:

```bash
ros2 topic pub --once /joy sensor_msgs/msg/Joy \
  "{axes: [0,0,0,0,0,0,0,0], buttons: [0,1,0,0,0,0,0,0,0,0,0]}"
```

Sets `buttons[1] = 1` and triggers the transition from *laying down* to *standing up*.

### Activate the policy

Published continuously at 20 Hz:

```bash
ros2 topic pub -r 20 /joy sensor_msgs/msg/Joy \
  "{axes: [0,0,0,0,0,0,0,0], buttons: [0,0,0,0,1,1,0,0,0,0,0]}"
```

Sets `buttons[4] = 1` and `buttons[5] = 1` (button combination). The command has to be repeated, so keep the publisher running until the state has switched, then stop it with `Ctrl+C`.

---

## Typical workflow

1. Start the simulator, the low-level controller, the policy node and the joy node.
2. Send the **stand up** command — the robot should reach a stable standing pose.
3. Send the **activate policy** command — `low_level_ctrl` switches to policy mode and starts tracking `/rl/target_pos`.
4. Drive the robot with the gamepad sticks (velocity commands).

---

## Notes on the real robot

* Launch the same nodes with `is_simulation:=false` / `--is_simulation False` and skip `mujoco_simulator.py`.
* The Go2's built-in **Sport Mode has to be disabled before every run** — it is not persistent across reboots or power cycles. If it is still active, it publishes on `/lowcmd` alongside the FSM node and the legs will shake. Verify with:

```bash
ros2 topic info /lowcmd --verbose
```

The publisher count should be **1**. If it is 2, Sport Mode is still running.
