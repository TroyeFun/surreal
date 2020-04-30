import os
import psutil
from surreal.main.ppo_configs import PPOLauncher
from surreal.test_helpers import integration_test


if __name__ == '__main__':
    print('BEGIN DDPG-Gym TEST')
    integration_test('/tmp/surreal',
                     os.path.join(os.path.dirname(__file__),
                                  '../surreal/main/ppo_configs.py'),
                     PPOLauncher(),
                     'robosuite:SawyerPickPlace')
    print('PASSED')
    self = psutil.Process()
    self.kill()
