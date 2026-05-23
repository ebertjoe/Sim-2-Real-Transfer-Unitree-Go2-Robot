#include <deploy_real/low_level_ctrl.hpp>

static double get_diff_norm(const vector<double> &v1, const vector<double> &v2)
{
    double s = 0;
    for (size_t i = 0; i < v1.size(); i++) s += pow(v1[i]-v2[i], 2);
    return sqrt(s);
}

static double get_norm(const vector<double> &v)
{
    double s = 0;
    for (double x : v) s += x*x;
    return sqrt(s);
}

LowLevelControl::LowLevelControl() : Node("finite_state_machine_node")
{
    this->declare_parameter("is_simulation", true);
    this->get_parameter("is_simulation", is_simulation);

    if (is_simulation) {
        RCLCPP_INFO(this->get_logger(), "Running in simulation mode.");
        cmd_puber_   = this->create_publisher<unitree_go::msg::LowCmd>("/mujoco/lowcmd", 10);
        state_suber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/mujoco/lowstate", 10,
            std::bind(&LowLevelControl::state_callback, this, std::placeholders::_1));
    } else {
        RCLCPP_INFO(this->get_logger(), "Running in real mode.");
        cmd_puber_   = this->create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);
        state_suber_ = this->create_subscription<unitree_go::msg::LowState>(
            "/lowstate", 10,
            std::bind(&LowLevelControl::state_callback, this, std::placeholders::_1));
    }

    target_pos_puber_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/pos", 10);

    init_cmd();

    target_pos_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/rl/target_pos", 10,
        std::bind(&LowLevelControl::target_pos_callback, this, std::placeholders::_1));

    joy_suber_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", 10,
        std::bind(&LowLevelControl::joy_callback, this, std::placeholders::_1));

    torque_suber_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/mujoco/torque", 10,
        std::bind(&LowLevelControl::torque_callback, this, std::placeholders::_1));

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(5),
        std::bind(&LowLevelControl::state_machine, this));
}

void LowLevelControl::init_cmd()
{
    // Stand-up / lay-down transition gains.
    // Stiffer than policy gains so the robot can actually reach the target
    // standing pose (thigh=1.0 on rear legs) against gravity.
    // These are ONLY used during the B-button transition — run_policy() uses
    // kp=25/kd=0.5 from deploy.yaml.
    kp = {40, 60, 60,   40, 60, 60,   40, 60, 60,   40, 60, 60};
    kd = { 2,  2,  2,    2,  2,  2,    2,  2,  2,    2,  2,  2};

    for (int i = 0; i < 20; i++) {
        cmd_msg_.motor_cmd[i].mode = 0x01;
        cmd_msg_.motor_cmd[i].q    = PosStopF;
        cmd_msg_.motor_cmd[i].kp   = 0;
        cmd_msg_.motor_cmd[i].dq   = VelStopF;
        cmd_msg_.motor_cmd[i].kd   = 0;
        cmd_msg_.motor_cmd[i].tau  = 0;
    }
    pos_data_ = std_msgs::msg::Float32MultiArray();
}

void LowLevelControl::state_callback(unitree_go::msg::LowState::SharedPtr msg)
{
    for (int i = 0; i < 12; i++)
        motor[i] = msg->motor_state[i];
}

void LowLevelControl::torque_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (static_cast<int>(msg->data.size()) >= 12)
        for (int i = 0; i < 12; i++)
            joint_torques_[i] = msg->data[i];
}

void LowLevelControl::target_pos_callback(std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    recieved_data_      = true;
    rl_target_pos_.data = msg->data;
    if (is_standing_ && should_run_policy_)
        run_policy();
}

void LowLevelControl::joy_callback(sensor_msgs::msg::Joy::SharedPtr msg)
{
    if (is_laydown_ && msg->buttons[1]) {
        should_stand_      = true;
        should_laydown_    = false;
        should_run_policy_ = false;
    } else if (is_standing_ && msg->buttons[0]) {
        should_laydown_    = true;
        should_stand_      = false;
        should_run_policy_ = false;
    } else if (is_standing_ && msg->buttons[4] && msg->buttons[5]) {
        should_laydown_    = false;
        should_stand_      = false;
        should_run_policy_ = true;
    }
    if (msg->axes[2] == -1 && msg->axes[5] == -1)
        rclcpp::shutdown();
}

void LowLevelControl::state_machine()
{
    state_obs();

    if (is_uncontrolled_ && !is_laydown_)
        state_transform(laydown_angels_);
    else if (is_laydown_ && should_stand_)
        state_transform(standing_angels_);
    else if (is_standing_ && should_laydown_)
        state_transform(laydown_angels_);
    else if (is_standing_ && should_run_policy_) {
        // policy runs in target_pos_callback
    } else {
        vector<double> &ta = is_standing_ ? standing_angels_ : laydown_angels_;
        vector<float> vec;
        for (int i = 0; i < 12; i++) {
            cmd_msg_.motor_cmd[i].mode = 0x01;
            cmd_msg_.motor_cmd[i].q    = ta[i];
            cmd_msg_.motor_cmd[i].kp   = kp[i];
            cmd_msg_.motor_cmd[i].dq   = 0;
            cmd_msg_.motor_cmd[i].kd   = kd[i];
            cmd_msg_.motor_cmd[i].tau  = 0;
            vec.push_back(static_cast<float>(ta[i]));
        }
        get_crc(cmd_msg_);
        cmd_puber_->publish(cmd_msg_);
        pos_data_.data = vec;
        target_pos_puber_->publish(pos_data_);
    }
}

void LowLevelControl::run_policy()
{
    if (!recieved_data_) {
        cmd_puber_->publish(cmd_msg_);
        return;
    }

    // deploy.yaml: stiffness=25, damping=0.5
    // These are the gains the policy was trained with — do NOT change.
    vector<float> vec;
    for (int i = 0; i < 12; i++) {
        cmd_msg_.motor_cmd[i].mode = 0x01;
        cmd_msg_.motor_cmd[i].q    = rl_target_pos_.data[i];
        cmd_msg_.motor_cmd[i].kp   = 25.0;
        cmd_msg_.motor_cmd[i].kd   = 0.5;
        cmd_msg_.motor_cmd[i].dq   = 0;
        cmd_msg_.motor_cmd[i].tau  = 0;
        vec.push_back(static_cast<float>(rl_target_pos_.data[i]));
    }
    get_crc(cmd_msg_);
    cmd_puber_->publish(cmd_msg_);
    pos_data_.data = vec;
    target_pos_puber_->publish(pos_data_);
}

void LowLevelControl::state_obs()
{
    vector<double> q(12), dq(12);
    for (int i = 0; i < 12; i++) { q[i] = motor[i].q; dq[i] = motor[i].dq; }

    if (get_diff_norm(q, laydown_angels_) < 0.25 && get_norm(dq) < 0.15) {
        if (!is_laydown_) { motion_time_ = 0; rate_count_ = 0; }
        is_laydown_ = true; is_uncontrolled_ = false; is_standing_ = false;
    } else if (get_diff_norm(q, standing_angels_) < 0.5 && get_norm(dq) < 0.15) {
        // Threshold 0.5 (not 0.3) — with kp=40/60/60 the robot reaches very
        // close to the target, but 0.5 gives tolerance for MuJoCo PD vs
        // Isaac Lab UnitreeActuator differences.
        if (!is_standing_) { motion_time_ = 0; rate_count_ = 0; }
        is_standing_ = true; is_laydown_ = false; is_uncontrolled_ = false;
    }
}

void LowLevelControl::state_transform(vector<double> &target_angels)
{
    motion_time_++;
    if (motion_time_ >= 0 && motion_time_ < 20)
        for (int i = 0; i < 12; i++) q_init_[i] = motor[i].q;

    if (motion_time_ >= 20) {
        rate_count_++;
        double rate = rate_count_ / 400.0;
        vector<float> vec;
        for (int i = 0; i < 12; i++) {
            q_des_[i] = jointLinearInterpolation(q_init_[i], target_angels[i], rate);
            cmd_msg_.motor_cmd[i].mode = 0x01;
            cmd_msg_.motor_cmd[i].q    = q_des_[i];
            cmd_msg_.motor_cmd[i].kp   = kp[i];
            cmd_msg_.motor_cmd[i].dq   = 0;
            cmd_msg_.motor_cmd[i].kd   = kd[i];
            cmd_msg_.motor_cmd[i].tau  = 0;
            vec.push_back(static_cast<float>(q_des_[i]));
        }
        get_crc(cmd_msg_);
        cmd_puber_->publish(cmd_msg_);
        pos_data_.data = vec;
        target_pos_puber_->publish(pos_data_);
    }
}

double LowLevelControl::jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    rate = std::min(std::max(rate, 0.0), 1.0);
    return initPos*(1-rate) + targetPos*rate;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LowLevelControl>());
    rclcpp::shutdown();
    return 0;
}