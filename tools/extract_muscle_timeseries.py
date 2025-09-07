#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, json, csv
import numpy as np
import xml.etree.ElementTree as ET
import torch
import mujoco

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from src.agents import agent_dict

# ---------- XML helpers ----------
def extract_muscle_names_from_xml(xml_path):
    import xml.etree.ElementTree as ET
    mus = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 1) 直接的 <actuator><muscle name="..."/></actuator>
        for n in root.findall(".//actuator/muscle"):
            name = n.get("name")
            if name:
                mus.append(name)

        # 2) MuJoCo 通用执行器被标注为肌肉的两种写法
        for n in root.findall('.//actuator/general[@class="muscle"]'):
            name = n.get("name")
            if name:
                mus.append(name)
        for n in root.findall('.//actuator/general[@dyntype="muscle"]'):
            name = n.get("name")
            if name:
                mus.append(name)

        # 去重、去 exo
        mus = sorted({m for m in mus if "exo" not in m.lower()})
    except Exception as e:
        print(f"[WARN] XML parse failed: {e}")
        mus = []
    return mus


def actuator_names(env):
    out = []
    for i in range(env.mj_model.nu):
        n = mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        out.append(n if isinstance(n, str) else str(n))
    return out

def map_names_to_ids(env, wanted_names):
    alln = actuator_names(env)
    name2id = {nm:i for i, nm in enumerate(alln)}
    kept, ids = [], []
    for nm in wanted_names:
        if nm in name2id:
            kept.append(nm); ids.append(name2id[nm])
    return kept, np.asarray(ids, np.int32)

# ---------- Hydra cfg ----------
def build_overrides(exp_name, xml_path, motion_file, init_pose, actor_type,
                    use_exo_in_action, exo_zero, headless, motion_id,
                    exo_actuators):
    """
    生成 Hydra overrides：
    - 小写 true/false 传布尔；
    - 用 '++' 保证 run.use_exo_in_action / run.exo_zero / run.exo_actuators 一定存在；
    - 显式覆写 learning.actor_type；
    """
    b = lambda x: "true" if x else "false"
    # 将 exo 执行器名用引号包起来，确保被当作字符串解析
    exo_list = ",".join([f"'{n}'" for n in exo_actuators])

    ov = [
        "env=env_im_eval",
        "learning=im_mlp",
        "run=eval_run",
        f"exp_name={exp_name}",
        "epoch=-1",

        f"run.headless={'True' if headless else 'False'}",
        "run.im_eval=True", "run.test=True",
        f"run.motion_file={motion_file}",
        f"run.initial_pose_file={init_pose}",
        "run.num_motions=1",
        "run.random_start=False",
        f"run.xml_path={xml_path}",

        f"learning.actor_type={actor_type}",

        f"++run.use_exo_in_action={b(use_exo_in_action)}",
        f"++run.exo_zero={b(exo_zero)}",
        f"++run.exo_actuators=[{exo_list}]",       # 关键：把 exo 名字写进 cfg

        "++run.exo_eval_both=false",
        f"+run.motion_ids=[{motion_id}]",
    ]
    return ov


def make_agent(cfg):
    device = torch.device("cpu")
    dtype  = torch.float32
    return agent_dict[cfg.learning.agent_name](cfg, dtype=dtype, device=device,
                                               training=False, checkpoint_epoch=cfg.epoch)

# ---------- rollout ----------
@torch.no_grad()
def rollout_and_export(agent, xml_path, out_dir):
    env = agent.env

    # 进入评估模式（很多项目在这里准备 motion 库等）
    env.start_eval(im_eval=True)

    # 关键：预先加载动作库，避免 reset() 内部用到未初始化的 motion_lib
    if hasattr(env, "sample_motions"):
        try:
            env.sample_motions()
        except Exception as e:
            print(f"[WARN] env.sample_motions() failed: {e}")

    # 解析 XML 肌肉名，并映射到当前 env 的 actuator 索引（自动剔除 exo）
    mus_names_xml = extract_muscle_names_from_xml(xml_path)
    if not mus_names_xml:
        print("[WARN] No muscle names parsed from XML. Falling back to all non-exo actuators.")
        alln = actuator_names(env)
        mus_names_xml = [n for n in alln if "exo" not in n.lower()]
    mus_names, mus_ids = map_names_to_ids(env, mus_names_xml)

    # 时间步长 dt
    dt = getattr(env, "dt", None)
    if dt is None:
        sim_inv = float(getattr(env, "sim_timestep_inv", getattr(env.cfg.env, "sim_timestep_inv", 150)))
        ctrl_inv = float(getattr(env, "control_frequency_inv", getattr(env.cfg.env, "control_frequency_inv", 5)))
        dt = ctrl_inv / sim_inv

    # reset
    obs, _ = env.reset()

    # 这里 env 已 reset
    all_act = actuator_names(env)
    print(f"[DEBUG] available actuators (first 10): {all_act[:10]} (total={len(all_act)})")

    # 如果没有找到 exo_idx，尝试用 XML 自动探测 exo 名字（以 exo_ 开头）
    if getattr(env, "exo_idx", None) is not None and len(env.exo_idx) == 0:
        guess = [n for n in all_act if n.startswith("exo_")]
        if guess:
            print(f"[WARN] exo_idx empty. Detected exo-like actuators in model: {guess}. "
                f"Try running with: --exo_actuators {','.join(guess)}")



    # 将 obs 摊平为 numpy
    def ext(o):
        if isinstance(o, dict):
            vals = []
            for v in o.values():
                try:
                    vals.append(np.asarray(v, np.float32).ravel())
                except:
                    pass
            return np.concatenate(vals).astype(np.float32)
        return np.asarray(o, np.float32)

    x = ext(obs)

    # 记录容器
    TMAX = 100000
    t_list, act_list, frc_list, pwr_list, pwr_pos_list, pwr_neg_list = [], [], [], [], [], []
    exo_tau_list, exo_frc_list = [], []  # NEW: EXO 力矩（ctrl）与 EXO 力（actuator_force）

    step = 0

    while True:
        # === 取“均值动作/确定性动作”：补 batch 维，兼容 MOE 的 gate softmax(dim=1) ===
        xt = torch.from_numpy(x).to(agent.dtype)
        if xt.dim() == 1:
            xt = xt.unsqueeze(0)   # [1, D]
        sel = agent.policy_net.select_action(xt, True)  # 第二个位置参数=True 表示确定性/均值
        a_t = sel[0] if isinstance(sel, (tuple, list)) else sel  # [1, A] 或 [A]
        if hasattr(a_t, "dim") and a_t.dim() > 1:
            a_t = a_t[0]  # 去掉 batch 维 -> [A]
        a = a_t.detach().cpu().numpy() if hasattr(a_t, "detach") else np.asarray(a_t)

        # 环境前向
        obs2, r, term, trunc, _ = env.step(agent.preprocess_actions(a))
        x = ext(obs2)

        # 采集执行器量
        ctrl = env.mj_data.ctrl.astype(np.float32)                    # [nu] —— 视作肌肉激活（AU）
        frc  = env.mj_data.actuator_force.astype(np.float32)          # [nu] —— 肌肉拉力（N）
        # 记录 EXO 通道（若存在）
        if getattr(env, "exo_idx", None) is not None and len(env.exo_idx) > 0:
            exo_tau_list.append(ctrl[env.exo_idx].copy())     # 力矩（或控制信号）
            exo_frc_list.append(frc[env.exo_idx].copy())      # 力（N）
            # 前几步打印均值，exo_off 时应接近 0
            if step < 3:
                print(f"[DEBUG] step {step} | mean exo_tau={np.mean(exo_tau_list[-1]):.6e}, "
                    f"mean exo_force={np.mean(exo_frc_list[-1]):.6e}")




        # actuator_velocity 可能在部分 MuJoCo 版本不存在；做稳健兜底
        v = getattr(env.mj_data, "actuator_velocity", None)
        if v is None:
            # 兜底：用 NaN 占位（分析时会自动跳过），并只报一次警告
            if step == 0:
                print("[WARN] mj_data.actuator_velocity not available; muscle power will be NaN.")
            p = np.full_like(frc, np.nan, dtype=np.float32)
        else:
            vel = np.asarray(v, dtype=np.float32)                     # [nu] —— 执行器速度（units/s）
            p = frc * vel                                             # [nu] —— 即时机械功率（W）

        # 只保留“肌肉”那几列
        t_list.append(step * dt); step += 1
        act_list.append(ctrl[mus_ids].copy())
        frc_list.append(frc[mus_ids].copy())
        p_mus = p[mus_ids].copy()
        pwr_list.append(p_mus)
        # 正/负功率分解（便于后续对“发力/吸收”统计）
        pwr_pos_list.append(np.clip(p_mus, 0, None))
        pwr_neg_list.append(np.clip(p_mus, None, 0))

        if term or trunc:
            break
        if step >= TMAX:
            break

    # 堆叠
    t = np.asarray(t_list, np.float32)
    if len(act_list) == 0:
        # 空轨迹兜底，避免后续写 CSV 404
        acts = np.zeros((1, len(mus_ids)), np.float32)
        frcs = np.zeros_like(acts)
        pwrs = np.zeros_like(acts)
        pwrs_pos = np.zeros_like(acts)
        pwrs_neg = np.zeros_like(acts)
    else:
        acts = np.vstack(act_list)           # [T, M]
        frcs = np.vstack(frc_list)           # [T, M]
        pwrs = np.vstack(pwr_list)           # [T, M]
        pwrs_pos = np.vstack(pwr_pos_list)   # [T, M]
        pwrs_neg = np.vstack(pwr_neg_list)   # [T, M]
    
    if len(exo_tau_list) > 0:
        exo_tau = np.vstack(exo_tau_list)   # [T, E]
        exo_frc = np.vstack(exo_frc_list)   # [T, E]
    else:
        exo_tau = None
        exo_frc = None



    os.makedirs(out_dir, exist_ok=True)

    # 写 CSV：第一列 time，其余为肌肉名
    def save_csv(path, mat, header_names):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time"] + header_names)
            for i in range(len(t)):
                row = [f"{t[i]:.6f}"] + [f"{v:.6f}" if np.isfinite(v) else "" for v in mat[i]]
                w.writerow(row)


    act_csv = os.path.join(out_dir, "muscle_activations.csv")
    frc_csv = os.path.join(out_dir, "muscle_forces.csv")
    pwr_csv = os.path.join(out_dir, "muscle_power.csv")          # 签名功率（W）
    pwrp_csv = os.path.join(out_dir, "muscle_power_pos.csv")     # 正功率部分（W）
    pwrn_csv = os.path.join(out_dir, "muscle_power_neg.csv")     # 负功率部分（W）
    exo_tau_csv = os.path.join(out_dir, "exo_torque.csv")
    exo_frc_csv = os.path.join(out_dir, "exo_force.csv")


    save_csv(act_csv, acts, mus_names)
    save_csv(frc_csv, frcs, mus_names)
    save_csv(pwr_csv, pwrs, mus_names)
    save_csv(pwrp_csv, pwrs_pos, mus_names)
    save_csv(pwrn_csv, pwrs_neg, mus_names)
    # 若存在 EXO 通道，再写 EXO CSV
    if exo_tau is not None:
    # EXO 通道的名字：直接用 env.exo_idx 映射到全体执行器名
        all_names = actuator_names(env)
        exo_names = [all_names[i] for i in env.exo_idx]
        save_csv(exo_tau_csv, exo_tau, exo_names)
        save_csv(exo_frc_csv, exo_frc, exo_names)



    # 元数据
    meta = dict(
        exo_idx=list(map(int, getattr(env, 'exo_idx', []))),
        has_exo=bool(getattr(env, 'exo_idx', None) is not None and len(env.exo_idx) > 0),
        exp_name=agent.cfg.exp_name,
        xml_path=xml_path,
        motion_file=str(agent.cfg.run.motion_file),
        motion_id=getattr(agent.cfg.run, "motion_id", None),
        actor_type=str(agent.cfg.learning.actor_type),
        use_exo_in_action=bool(agent.cfg.run.use_exo_in_action),
        exo_zero=bool(getattr(agent.cfg.run, "exo_zero", False)),
        dt=float(dt),
        muscle_names=mus_names,
        num_steps=int(len(t)),
        fields=dict(
            activation="AU (unitless)",
            force="N",
            power="W",
            power_pos="W (clip+)",
            power_neg="W (clip-)"
        ),
        actuator_velocity_available=bool(getattr(env.mj_data, "actuator_velocity", None) is not None)
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 打印汇总
    print(f"[OK] Saved CSV to: {out_dir}")
    print(" - muscle_activations.csv")
    print(" - muscle_forces.csv")
    print(" - muscle_power.csv")
    print(" - muscle_power_pos.csv")
    print(" - muscle_power_neg.csv")
    if exo_tau is not None:
        print(" - exo_torque.csv")
        print(" - exo_force.csv")
    print(" - meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", required=True)
    ap.add_argument("--xml_path", required=True)
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--initial_pose_file", required=True)
    ap.add_argument("--motion_id", type=int, default=0)
    ap.add_argument("--actor_type", default="gauss")  # 或 "gauss"
    to_bool = lambda s: str(s).lower() in {"1", "t", "true", "y", "yes"}
    ap.add_argument("--use_exo_in_action", type=to_bool, default=False)
    ap.add_argument("--exo_zero",         type=to_bool, default=True)
    ap.add_argument("--headless",         type=to_bool, default=True)
    ap.add_argument("--out_dir", required=True, help="Directory to save CSVs")
    # NEW: 指定 EXO 执行器名称，逗号分隔
    ap.add_argument("--exo_actuators", type=str, default="exo_knee_r,exo_knee_l",
                    help="Comma-separated exo actuator names as in XML")
    args = ap.parse_args()
    exo_names = [s.strip() for s in args.exo_actuators.split(",") if s.strip()]

    # 组装 overrides
    overrides = build_overrides(
        exp_name=args.exp_name,
        xml_path=args.xml_path,
        motion_file=args.motion_file,
        init_pose=args.initial_pose_file,
        actor_type=args.actor_type,
        use_exo_in_action=args.use_exo_in_action,
        exo_zero=args.exo_zero,
        headless=args.headless,
        motion_id=args.motion_id,
        exo_actuators=exo_names,             # <= NEW
    )


    # 只初始化一次 Hydra，然后 compose 一次
    cfg_dir = os.path.join(REPO, "cfg")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # 构造 agent 并导出
    agent = make_agent(cfg)

    # 兜底注入 exo_zero（防止环境内部没读到 cfg）
    if hasattr(agent, "env"):
        if hasattr(agent.env, "set_exo_zero"):
            agent.env.set_exo_zero(bool(args.exo_zero))
        else:
            setattr(agent.env, "_exo_zero", bool(args.exo_zero))

    # 更准确的调试打印：直接读 cfg
    print(f"[DEBUG] cfg.run.use_exo_in_action={bool(getattr(agent.cfg.run, 'use_exo_in_action', False))}")
    print(f"[DEBUG] cfg.run.exo_actuators={getattr(agent.cfg.run, 'exo_actuators', None)}")

    rollout_and_export(agent, args.xml_path, args.out_dir)

if __name__ == "__main__":
    main()
