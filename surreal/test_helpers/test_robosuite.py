import os
import psutil
from surreal.main.ppo_configs import PPOLauncher
from surreal.main.ddpg_configs import DDPGLauncher
from surreal.test_helpers import integration_test


if __name__ == '__main__':
    print('BEGIN DDPG-Gym TEST')
    launcher = DDPGLauncher()
    integration_test('/tmp/surreal',
                     os.path.join(os.path.dirname(__file__),
                                  '../main/ddpg_configs.py'),
                     launcher,
                     'robosuite:SawyerPickPlaceMultiTaskTarget')
    print('PASSED')
    self = psutil.Process()
    self.kill()
