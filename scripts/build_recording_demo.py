"""Build deterministic, explicitly synthetic inputs for the product walkthrough.

No test outcomes or training labels are generated. The twelve candidates share
one corridor, endpoints and vehicle; geometry and speeds vary independently.
"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np


def build(root):
    root = Path(root)
    routes = root / '01_上传路线'
    invalid = root / '03_异常输入_不要混入正常任务'
    routes.mkdir(parents=True, exist_ok=True)
    invalid.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0., 160., 801)
    # Identical road geometry for every candidate: a smooth, symmetric narrowing.
    half_width = 4. - 1.2 * np.exp(-((x - 85.) / 18.) ** 2)
    left = np.column_stack([x, half_width, np.zeros_like(x)])
    right = np.column_stack([x, -half_width, np.zeros_like(x)])
    config = dict(vehicle_structure='five_axis', map_id='synthetic_corridor_v1',
                  length_m=12., width_m=3.2, reference_from_rear_m=6.,
                  dimension_source='illustrative_demo_values',
                  data_origin='synthetic_functional_rehearsal',
                  description='自制功能演练；假设五轴配置，非真实车型和现场测绘。')
    (root / '02_车辆配置_假设五轴.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = []
    for i, (offset, label) in enumerate([(0., '居中'), (.5, '小幅左移'), (-.5, '小幅右移'),
                                        (1., '左侧靠近'), (-1., '右侧靠近'), (1.6, '明显左偏')]):
        y = offset * np.sin(np.pi * x / 160.) ** 2
        s = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
        yaw = np.unwrap(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
        kappa = np.gradient(yaw, s)
        for speed, speed_label in [(2., '低速'), (5., '较高速')]:
            v = speed * (1. - .15 * np.sin(np.pi * x / 160.) ** 2)
            arrays = dict(x_m=x, y_m=y, s_m=s, z_m=np.zeros_like(x),
                          yaw_rad=yaw, kappa_1pm=kappa, dkappa_ds_1pm2=np.gradient(kappa, s),
                          v_profile_mps=v, yaw_rate_ref_radps=kappa * v,
                          left_clearance_m=half_width-y, right_clearance_m=half_width+y,
                          left_boundary_xyz_m=left, right_boundary_xyz_m=right,
                          # Assumed steering proxy for input exercises, not a validated multi-axle controller.
                          steer_ff_rad=np.arctan(kappa[:, None] * np.linspace(4., -4., 10)[None, :]))
            name = f'{len(manifest)+1:02d}_自制_{label}_{speed_label}.npz'
            np.savez_compressed(routes / name, **arrays)
            manifest.append(dict(filename=name, offset_m=offset, nominal_speed_mps=speed,
                                 sha256=hashlib.sha256((routes/name).read_bytes()).hexdigest(),
                                 origin='synthetic', label=None, points=801))
            if len(manifest) == 1:
                broken = {k:v for k,v in arrays.items() if k != 'v_profile_mps'}
                np.savez_compressed(invalid / '缺少速度字段_应拒绝.npz', **broken)
    (root / '生成依据与文件清单.json').write_text(json.dumps(dict(
        scene='自制收窄道路', environment_version='synthetic-corridor-v1',
        corridor='x=0..160m; y=±(4-1.2*exp(-((x-85)/18)^2)); z=0',
        candidates='y=offset*sin(pi*x/160)^2; all share (0,0) and (160,0)',
        steering='atan(kappa * linspace(4,-4,10)); hypothetical input proxy only',
        purpose='功能演练，非泛化测试、非真实标签、非现场工程案例', files=manifest), ensure_ascii=False, indent=2), encoding='utf-8')
    (root / '04_演练检查记录_待验证.txt').write_text(
        'PathGuard 自制文件导入演练记录\n\n'
        '材料性质：软件功能演练附件，不是仿真通过证明或实车测试报告。\n'
        '任务：自制收窄道路候选方案演练\n环境版本：synthetic-corridor-v1\n'
        '输入：12条合成候选路线，同一道路和起终点；假设五轴车长12m、宽3.2m。\n'
        '检查范围：文件能否导入、形成排序、展示空间试算、保存记录并导出任务包。\n'
        '工程通行结论：待验证。未执行闭环仿真或现场测试。\n'
        '建议表单：验证结果=待验证；验证方式=人工复核；结论复核状态=提交者填写。\n'
        '训练处理：保持待审核，不能据此加入真实标签训练集。\n', encoding='utf-8')
    (root / 'README.md').write_text(
        '# 自制收窄道路演练包\n\n'
        '这是一套可复现的功能演练输入，不是已有冻结样本、真实车型数据或测试标签。\n\n'
        '1. 解压后在“风险工作台→新建评估任务”填写任务名、场景、环境版本 synthetic-corridor-v1。\n'
        '2. 输入状态选择“待评估”。多选 01_上传路线 下全部12个NPZ，配置上传 02_车辆配置_假设五轴.json。\n'
        '3. 点击“保存任务并评估”，依次展示任务概览、验证顺序、路线详情。\n'
        '4. 在路线详情展开专业指标/几何设置，启用假设尺寸边界余量试算。比较居中与明显左偏路线，再改宽度。\n'
        '5. 登记“待验证”的人工复核记录，附上04文本，并导出任务证据包。\n'
        '6. 异常NPZ必须另外创建任务导入，应提示缺少速度字段。不要混入正常12条。\n\n'
        '12条来自6个横向方案×2个计划速度；12不是模型训练规模，也不表示12条独立真实任务。\n'
        '道路最窄5.6m，所有候选共享完全相同边界。车长、车宽改变几何试算，不直接改55维模型分数。\n'
        '模型分数以实际输出为准，不承诺合成方案单调排序，不据此报告准确率。\n', encoding='utf-8')
    archive = root.parent / 'PathGuard自制文件演示包.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(root))
    return archive


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    print(build(p.parse_args().output))
