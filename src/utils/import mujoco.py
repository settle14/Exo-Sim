import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path('/home/extra/my_projects/Kinesis/data/xml/myolegs_x2.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Hit ESC to exit viewer.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
