``` python
class Env:
    """
    We use _FLU_ coordinate for the environment.

    Obstacles definition:
        balls:                       [center x, center y, center z, radius]
        cyl (cylinder):              [center x, center y, radius]
        cyl_h (horizontal cylinder): [center x, center z, radius]
        voxel:                       [center x, center y, center z, edge length x, edge length y, edge length z]
    """

    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, speed_mtp=1,
                 cam_angle=10, random_rotation=False, rgb=False) -> None:
        """Args:
            batch_size
            width: canvas width (px)
            height: canvas height (px)
            grad_decay: gradient decay half life
            device
            fov_x_half_tan: half of the tangent horizontal camera field of view, i.e., tan(horizontal fov) / 2
            single: single agent mode
            gate: generate a gate in the middle of the environment
            ground_voxels: generate large voxels on the ground (to simulate buildings)
            speed_mtp: multiplier for max speed. The training max speed is in [0.75 * speed_mtp, 5.75 * speed_mtp].
            cam_angle: camera mounting pitch angle
            random_rotation: randomly rotate environment (tilt voxels)
            rgb: simulate depth estimation noises
        """
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        self.rgb = rgb
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan
        self.fov_y_half_tan = fov_x_half_tan / width * height

        # The obstacles are uniformly scattered in a box. We define weight (w) and bias (b), and generate using random() * w + b.
        self.rgb_error = torch.randn((batch_size, 1, 1), device=device)
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([6., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([1.0, -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.sub_dt = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)

        # Initial position
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-1.5,  1.,  1],
            [ 9.5,  1.,  1],
            [-1.5,  3.,  1],
            [ 9.5,  3.,  1],
            [-1.5, -1.,  1],
            [ 9.5, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]

        # Target position
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]

        # reserved for optical flow
        self.flow = torch.empty((batch_size, 2, height, width), device=device)
        self.mock_depth =  torch.ones((batch_size, height, width), device=device)
        self.force_inplace = False
        self.init_state_tensors = [None] * 6
        self.reset()
        self.init_state_tensors = self.act, self.p, self.v, self.a, self.R, self.dg

    def __setattr__(self, __name: str, __value):
        if getattr(self, 'force_inplace', False) and torch.is_tensor(getattr(self, __name, None)):
            self.__getattribute__(__name)[:] = __value
        else:
            super().__setattr__(__name, __value)

    def reset(self):
        self.force_inplace = False
        self.act, self.p, self.v, self.a, self.R, self.dg = self.init_state_tensors
        self.force_inplace = True
        """Reset and generate a random environment."""
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # env
        balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan = self.fov_x_half_tan
        self._fov_y_half_tan = self.fov_y_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        # change here
        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 5 * rd) * self.speed_mtp
        # self.max_speed = (1.5 + 4 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,)) < 0.4
        balls[~roof, :15, :2] = cyl[~roof, :15, :2]
        voxels[~roof, :15, :2] = cyl[~roof, 15:, :2]
        balls[~roof, :15] = balls[~roof, :15] + self.roof_add[:4]
        voxels[~roof, :15] = voxels[~roof, :15] + self.roof_add

        # clip obstacle within p_init and p_end
        balls[..., 0] = torch.minimum(torch.maximum(balls[..., 0], balls[..., 3] + 0.5 / scale), 8 - 0.5 / scale - balls[..., 3])
        voxels[..., 0] = torch.minimum(torch.maximum(voxels[..., 0], voxels[..., 3] + 0.5 / scale), 8 - 0.5 / scale - voxels[..., 3])
        cyl[..., 0] = torch.minimum(torch.maximum(cyl[..., 0], cyl[..., 2] + 0.5 / scale), 8 - 0.5 / scale - cyl[..., 2])
        cyl_h[..., 0] = torch.minimum(torch.maximum(cyl_h[..., 0], cyl_h[..., 2] + 0.5 / scale), 8 - 0.5 / scale - cyl_h[..., 2])
        voxels[roof, 0, 2] = voxels[roof, 0, 2] * 0.5 + 201
        voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r = 6.0 + torch.rand((B, 1), device=device) * 6
            ground_balls_r_ground = 2.0 + torch.rand((B, 1), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            balls[:, :1, 3] = ground_balls_r
            balls[:, :1, 2] = ground_balls_h - ground_balls_r - 1
            # balls[:, :1, 0] += 1.0
            

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            # ground_voxels[:, :, 0] += 1.0
            voxels = torch.cat([voxels, ground_voxels], 1)

        voxels[:, :, 1] *= (self.max_speed + 4) / scale
        balls[:, :, 1] *= (self.max_speed + 4) / scale
        cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, balls, cyl, cyl_h, voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)

            voxels = torch.cat([voxels, gate], 1)
        voxels[..., 0] *= scale 
        balls[..., 0] *= scale
        cyl[..., 0] *= scale 
        cyl_h[..., 0] *= scale 
        if self.ground_voxels:
            balls[:, :1, 0] = torch.minimum(torch.maximum(balls[:, :1, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # drone
        self.pitch_ctl_delay = 15 + 1.5 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) * 2 - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        # randomly rotate environment (tilt voxels)
        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            voxels[..., :3] = (R @ voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            balls[..., :3] = (R @ balls[..., :3].transpose(1, 2)).transpose(1, 2)
            cyl[..., :3] = (R @ cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_rotation(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        
        if  torch.isnan(self.R).any():
            print("self.R ", self.R)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.3 + 0.1

        # drag coef
        # self.drag_2 = torch.rand((B, 2), device=device) * 0.1 + 0.3
        self.drag_2 = torch.rand((B, 2), device=device) * 0.1 + 0.18
        self.drag_2[:, 0] *= 0.06
        # self.z_drag_coef = torch.ones((B, 1), device=device)
        self.z_drag_coef = torch.ones((B, 1), device=device) * 4.0 + torch.rand((B, 1), device=device) * 0.4

        # apply obstacle
        self.balls = balls
        self.voxels = voxels
        self.cyl = cyl
        self.cyl_h = cyl_h
        self.force_inplace = False
```


## 障碍物生成规则总结（旧项目 Env）

    以下总结基于上述 `Env.reset()` 中的生成逻辑（使用 FLU 坐标系）：

    ### 统一约定
    - 采样方式：均匀随机 `rand() * w + b`。
    - 批大小记为 B。
    - 速度相关缩放：
        - `rd ~ U(0,1)`, `max_speed = (0.75 + 5 * rd) * speed_mtp`（按组重复）；
        - `scale = clamp_min(max_speed - 0.5, 1)`；
        - 后处理：`centers[..., 0] *= scale`（X 方向拉伸），以及 `centers[..., 1] *= (max_speed + 4) / scale`（Y 方向随速度扩展）。
    - X 方向边界裁剪（确保起点与终点之间留出通道）：
        - 球：`x ∈ [r + 0.5/scale, 8 - 0.5/scale - r]`；
        - 体素：`x ∈ [ex/2 + 0.5/scale, 8 - 0.5/scale - ex/2]`；
        - 竖直圆柱：`x ∈ [r + 0.5/scale, 8 - 0.5/scale - r]`；
        - 水平圆柱：`x ∈ [r + 0.5/scale, 8 - 0.5/scale - r]`。
    - 随机整体旋转（可选）：若开启 `random_rotation`，对环境（障碍物中心与边）施加绕 Z 的随机小角度旋转。

    ### 球体（sphere / balls）
    - 原始采样：`balls ∈ R^{B×30×4}`，权重与偏置：
        - `ball_w = [8., 18, 6, 0.2]`，`ball_b = [0., -9, -1, 0.4]`
        - 含义：
            - `x ∈ [0, 8]`
            - `y ∈ [-9, 9]`
            - `z ∈ [-1, 5]`
            - `r ∈ [0.4, 0.6]`
    - “屋顶”场景混合：
        - 抽样 `roof ~ Bernoulli(p=0.4)`；当 `~roof` 时，将前 15 个球的前两列 `(x,y)` 用对应 `cyl` 的 `(x,y)` 替换（与圆柱混合分布）。
        - 同时对前 15 个球加偏移 `roof_add[:4] = [0, 0, 2.5, 1.5]`（抬高 z 并增大 r）。
    - 后处理：按“统一约定”执行速度缩放与 X 方向裁剪。

    ### 体素（voxel / axis-aligned boxes）
    - 原始采样：`voxels ∈ R^{B×30×6}`，权重与偏置：
        - `voxel_w = [8., 18, 6, 0.1, 0.1, 0.1]`，`voxel_b = [0., -9, -1, 0.2, 0.2, 0.2]`
        - 含义：
            - 中心 `x ∈ [0, 8]`, `y ∈ [-9, 9]`, `z ∈ [-1, 5]`
            - 边长 `ex, ey, ez ∈ [0.2, 0.3]`
    - “屋顶”场景混合：
        - 当 `~roof` 时，将前 15 个体素的 `(x,y)` 用 `cyl` 的后 15 个 `(x,y)` 替换；
        - 并加偏移 `roof_add = [0, 0, 2.5, 1.5, 1.5, 1.5]`（抬高 z、放大边长）。
    - 特例：当 `roof` 为真时，对 `voxels[roof, 0]`（第一个体素）设置：`z = z*0.5 + 201`，`ex,ey,ez = 200`，相当于放置一个极远/极大的体素以“遮蔽”影响。
    - 地面体素（可选 `ground_voxels`）：
        - 额外采样 `ground_voxels ∈ R^{B×10×6}`，
        - `ground_voxel_w = [6., 18, 0, 2.9, 2.9, 1.9]`，`ground_voxel_b = [1.0, -9, -1, 0.1, 0.1, 0.1]`；
        - 之后设定 `center_z = ez - 1` 使其贴地；并与原 `voxels` 拼接。
    - 后处理：`y` 速度缩放、`x` 裁剪同“统一约定”。

    ### 竖直圆柱（cylinder_v / cyl）
    - 原始采样：`cyl ∈ R^{B×30×3}`，权重与偏置：
        - `cyl_w = [8., 18, 0.35]`，`cyl_b = [0., -9, 0.05]`
        - 含义：
            - `x ∈ [0, 8]`, `y ∈ [-9, 9]`, `r ∈ [0.05, 0.4]`（沿 z 方向延展）
    - 后处理：`y` 速度缩放、`x` 裁剪同“统一约定”。

    ### 水平圆柱（cylinder_h / cyl_h）
    - 原始采样：`cyl_h ∈ R^{B×2×3}`，权重与偏置：
        - `cyl_h_w = [8., 6, 0.1]`，`cyl_h_b = [0., 0, 0.05]`
        - 含义：
            - `x ∈ [0, 8]`, `z ∈ [0, 6]`, `r ∈ [0.05, 0.15]`（沿 y 方向延展）
    - 后处理：`x` 速度缩放与裁剪同“统一约定”。

    ### 门（可选 gate）
    - 若 `self.gate` 为真：
        - 先采样门圆心与半径：`gate ~ U*w + b`，其中 `gate_w = [2., 2, 1.0, 0.5]`, `gate_b = [3., -1, 0.0, 0.5]`；
        - 用 `find_nearest_pt(...)` 计算与现有障碍最邻近点，若过近则丢弃（`x = -50`）；
        - 将门用 4 块细长体素条组合成“方框”形态并追加到 `voxels`。

    ---

    ## 新环境（indoor.py）中的障碍物生成（设计与现状）

    文件：`examples/worldgen/indoor.py`

    ### 设计要点
    - 流水线：Room → Lights → Gates → Cameras → Obstacles。
    - 配置模型：
        - `Obstacles`：
            - `types`: 形状类型列表（默认包含 `sphere`, `box`, `cylinder`, `triangular_pyramid`, `square_pyramid`, `square_frustum`, `cone_frustum` 等）。
            - `count`: 目标数量（默认 80）。
            - `density`: 面积密度（每平米数量，默认 0.2）。
            - `path_ratio`: 预留通行路径倍数（默认 2.0）。
            - `size_variation`: 尺寸扰动范围（默认 `(0.5, 0.5, 0.125)`）。
        - `SquareGate` / `CircleGate` 设计了 `post_euler(robot_size)` 与 `aabb` 字段，意在按机器人尺寸和 `path_ratio` 计算朝向与通道包围盒。

    ### 当前实现状态
    - 房间与光照、门与相机的构建路径已完成（或接近完成）。
    - 障碍物相关：
        - `prealloc_obstacles()` 与 `build_obstacles()` 仍为 TODO；
        - `post_euler()`（门方向/包围盒计算）未完成；
        - 存在 `occupied_spaces` 机制与 `path_ratio` 意图，说明后续将采用空间占用/碰撞与通道留空的采样策略。

    ### 预期的随机生成方向（从代码意图推断）
    - 在房间 `extent` 内按 `density` 或 `count` 决定数量；
    - 从 `types` 中采样形状，依据 `size_variation` 产生尺寸；
    - 放置时检查与 `occupied_spaces` 不重叠，并留足 `path_ratio * robot_size` 的通行空间；
    - 形状将以实体网格或基本几何体添加至 `scene`，并可能绑定材质与碰撞组属性。

    ---

    ## 旧 vs 新：规则对比与差异

    - 采样域：
        - 旧：固定盒形域（通过 `w,b` 直接控制范围），与速度耦合（X 拉伸，Y 放大）。
        - 新：以实际房间 `extent` 为边界，预计与速度解耦，更物理一致。

    - 通道留空与约束：
        - 旧：仅在 X 方向按半径/半边长进行裁剪，留出 `[0.5/scale]` 的通道；
        - 新：计划使用 `occupied_spaces + path_ratio * robot_size` 的空间约束，通道更明确可控（任意方向）。

    - 形状与多样性：
        - 旧：sphere、voxel、竖直/水平圆柱，体素轴对齐；
        - 新：除上述基础外，还包含棱锥、台体、圆台等更多多面体/曲面体，材质/纹理更丰富。

    - 交互与构件：
        - 旧：门是用 4 个长盒近似；
        - 新：门使用真实网格（`square_gate.obj` / `circle_gate.obj`）及朝向控制，具备更真实的视觉与碰撞。

    - 实现成熟度：
        - 旧：完整可用，训练/仿真中广泛使用；
        - 新：架构与接口已搭好，障碍物生成核心逻辑待落地（TODO）。

    ### 建议的落地步骤（供参考）
    - 在 `build_obstacles()` 中实现：
        1) 按 `count` 或 `density*room_area` 确定目标数量；
        2) 逐个采样形状与尺寸，计算 AABB；
        3) 用 `occupied_spaces` 做 AABB 重叠检测；
        4) 结合 `path_ratio*robot_size` 留出主要通道（可从起点到终点的走廊或多段航线）；
        5) 将实例化的几何体/网格加入 `scene`，记录入 `occupied_spaces`；
        6) 提供调试开关，绘制占用框（`draw_debug_occupied_spaces`）。