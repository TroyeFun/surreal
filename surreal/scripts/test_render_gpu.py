import mujoco_py
model = mujoco_py.load_model_from_path('../../robosuite/robosuite/models/assets/arenas/table_arena.xml')
sim = mujoco_py.MjSim(model)
context = mujoco_py.MjRenderContext(sim)
