from ..config import extend_config
import os

# ======================== Agent-Learner side ========================
BASE_LEARNER_CONFIG = {
    'model': '_dict_',
    'algo': {
        # Agent class to instantiate
        # Learner class to instantiate
        'n_step': 1,
        'gamma': '_float_',
        'use_batchnorm': False,
        'limit_training_episode_length': 0,
        'network': {
            'actor_regularization': 0.0,
            'critic_regularization': 0.0,
        },
    },
    'replay': {
        # The replay class to instantiate
        'batch_size': '_int_',
        'replay_shards': 1,
    },
    'parameter_publish': {
        # Minimum amount of time (seconds) between two parameter publish
        'min_publish_interval': 0.3, 
    },
}


# ======================== Env side ========================
BASE_ENV_CONFIG = {
    'env_name' : '_str_',
    'sleep_time': 0.0,
    'video' : {
        'record_video' : False,
        'max_videos': 10,
        'record_every': 10,
        'save_folder': None,
    },
    'eval_mode': {}, # for providing different env init args when in eval
    'action_spec': {},
    'obs_spec': {},
    'frame_stacks': 1,
    'frame_stack_concatenate_on_env': True,
    # 'action_spec': {
    #     'dim': '_list_',
    #     'type': '_enum[continuous, discrete]_'
    # },
    # 'obs_spec': {
    #     'dim': '_list_',
    #     'type': ''  # TODO uint8 format
    # },
}


# ======================== Session side ========================
BASE_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'collector_frontend_host': '_str_',  # upstream from agents' pusher
        'collector_frontend_port': '_int_',
        'collector_backend_host': '_str_',  # upstream from agents' pusher
        'collector_backend_port': '_int_',
        'sampler_frontend_host': '_str_',  # downstream to Learner request
        'sampler_frontend_port': '_int_',
        'sampler_backend_host': '_str_',  # downstream to Learner request
        'sampler_backend_port': '_int_',
        'max_puller_queue': '_int_',  # replay side: pull queue size
        'evict_interval': '_float_',  # in seconds
        'tensorboard_display': True,  # display replay stats on Tensorboard
    },
    'sender': {
        'flush_iteration': '_int_',
        'flush_time': '_int_',
    },
    'ps': {
        'parameter_serving_frontend_host': '_str_',
        'parameter_serving_frontend_port': '_int_',
        'parameter_serving_backend_host': '_str_',
        'parameter_serving_backend_port': '_int_',
        'shards': '_int_',
        'publish_host': '_str',  # upstream from learner
        'publish_port': '_int_'
    },
    'tensorplex': {
        'host': '_str_',
        'port': '_int_',
        'tensorboard_port': '_int_',  # tensorboard port
        'agent_bin_size': 8,
        'max_processes': 4,
        'update_schedule': {  # TODO rename this to 'periodic'
            # for TensorplexWrapper:
            'training_env': '_int_',  # env record every N episodes
            'eval_env': '_int_',
            'eval_env_sleep': '_int_',  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': '_int_',  # agent.tensorplex.add_scalars()
            # WARN!!: DEPRECATED
            'learner': '_int_',  # learner.tensorplex.add_scalars()
            'learner_min_update_interval': '_int_', #Update tensorplex at most every ? seconds
        }
    },
    'loggerplex': {
        'host': '_str_',
        'port': '_int_',
        'overwrite': False,
        'level': 'info',
        'show_level': True,
        'time_format': 'dhm',
        'enable_local_logger': '_bool_',
        'local_logger_level': 'info',
        'local_logger_time_format': 'dhm',
        'local_logger_format': '[{asctime}][{filename}][line:{lineno}]',
    },
    'agent': {
        'fetch_parameter_mode': '_str_',
        'fetch_parameter_interval': int,
    },
    'learner': {
        'num_gpus': '_int_',
        'prefetch_host': '_str_',
        'prefetch_port': '_int_',
        'prefetch_processes': '_int_',
        'max_prefetch_queue': '_int_',  # learner side: max number of batches to prefetch
        'max_preprocess_queue': '_int_',  # learner side: max number of batches to preprocess
    },
    'checkpoint': {
        'restore': '_bool_',  # if False, ignore the other configs under 'restore'
        'restore_folder': None,  # if None, use the same session folder.
                            # Otherwise restore ckpt from another experiment dir.
        'learner': {
            'restore_target': '_int_',
            'mode': '_enum[best,history]_',
            'keep_history': '_int_',
            'keep_best': '_int_',
            'periodic': '_int_',
            'min_interval': '_int_',
        },
        'agent': {
            'restore_target': '_int_',
            'mode': '_enum[best,history]_',
            'keep_history': '_int_',
            'keep_best': '_int_',
            'periodic': '_int_',
        },
    }
}


LOCAL_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'collector_frontend_host': 'localhost',  # upstream from agents' pusher
        'collector_frontend_port': 8001,
        'collector_backend_host': 'localhost',  # upstream from agents' pusher
        'collector_backend_port': 8002,
        'sampler_frontend_host': 'localhost',  # downstream to Learner request
        'sampler_frontend_port': 8003,
        'sampler_backend_host': 'localhost',  # downstream to Learner request
        'sampler_backend_port': 8004,
        'max_puller_queue': 10000,  # replay side: pull queue size
        'evict_interval': 0.,  # in seconds
        'tensorboard_display': True,  # display replay stats on Tensorboard
    },
    'sender': {
        'flush_iteration': '_int_',
        'flush_time': 0,
    },
    'ps': {
        'parameter_serving_frontend_host': 'localhost',
        'parameter_serving_frontend_port': 8005,
        'parameter_serving_backend_host': 'localhost',
        'parameter_serving_backend_port': 8006,
        'shards': 2,
        'publish_host': 'localhost',  # upstream from learner
        'publish_port': 8007
    },
    'tensorplex': {
        'host': 'localhost',
        'port': 8008,
        'tensorboard_port': 6007,
        'update_schedule': { # TODO: rename this to 'periodic'
            # for TensorplexWrapper:
            'training_env': 20,  # env record every N episodes
            'eval_env': 20,
            'eval_env_sleep': 30,  # throttle eval by sleep n seconds
            # for manual updates:
            'agent': 20,  # agent.tensorplex.add_scalars()
            'learner': 20,  # learner.tensorplex.add_scalars()
            'learner_min_update_interval': 30, #Update tensorplex at most every 30 seconds
        }
    },
    'loggerplex': {
        'host': 'localhost',
        'port': 8009,
        'enable_local_logger': True,
    },
    'agent': {
        # fetch_parameter_mode: 'episode', 'episode:<n>', 'step', 'step:<n>'
        # every episode, every n episodes, every step, every n steps
        'fetch_parameter_mode': 'episode',
        'fetch_parameter_interval': 1,
    },
    'learner': {
        'num_gpus': 0,
        'prefetch_host': 'localhost',
        'prefetch_port': 8010,
        'prefetch_processes': 2,
        'max_prefetch_queue': 10,  # learner side: max number of batches to prefetch
        'max_preprocess_queue': 2,  # learner side: max number of batches to preprocess
    },
    'checkpoint': {
        'restore': False,  # if False, ignore the other configs under 'restore'
        'restore_folder': None,
        'learner': {
            'restore_target': 0,
            'mode': 'history',
            'keep_history': 20,
            'keep_best': 0, # TODO don't keep best unless we solve the learner score issue
            'periodic': 100000, # Save every 100000 steps
            'min_interval': 15 * 60, # No checkpoint less than 15 min apart.
        },
        'agent': {
            'restore_target': 0,
            'mode': 'history',
            'keep_history': 20,
            'keep_best': 0, # TODO don't keep best unless we solve the learner score issue
            'periodic': 100,
        },
    }
}

LOCAL_SESSION_CONFIG = extend_config(LOCAL_SESSION_CONFIG, BASE_SESSION_CONFIG)


KUBE_SESSION_CONFIG = {
    'folder': '_str_',

    'replay': {
        'collector_frontend_host': '_str_',  # upstream from agents' pusher
        'sampler_frontend_host': '_str_',  # downstream to Learner request
    },
    'sender': {
        'flush_iteration': '_int_',
    },
    'ps': {
        'host': '_str_',  # downstream to agent requests
        'publish_host': '_str_',  # upstream from learner
    },
    'tensorplex': {
        'host': '_str_',
    },
    'loggerplex': {
        'host': '_str_',
    },
}

KUBE_SESSION_CONFIG = extend_config(KUBE_SESSION_CONFIG, LOCAL_SESSION_CONFIG)


cuda_device = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
print('warning: setting port according to CUDA_VISIBLE_DEVICES', cuda_device)

SESSION_CONFIG_EXT = {
    'folder': '_str_',
    'replay': {
        'collector_frontend_port': 8001 + 1000 * cuda_device,
        'collector_backend_port': 8002 + 1000 * cuda_device,
        'sampler_frontend_port': 8003 + 1000 * cuda_device,
        'sampler_backend_port': 8004 + 1000 * cuda_device,
    },
    'sender': {
        'flush_iteration': '_int_',
    },
    'ps': {
        'parameter_serving_frontend_port': 8005 + 1000 * cuda_device,
        'parameter_serving_backend_port': 8006 + 1000 * cuda_device,
        'publish_port': 8007 + 1000 * cuda_device
    },
    'tensorplex': {
        'port': 8008 + 1000 * cuda_device,
        'tensorboard_port': 8011 + 1000 * cuda_device,
    },
    'loggerplex': {
        'port': 8009 + 1000 * cuda_device,
    },
    'learner': {
        'prefetch_port': 8010 + 1000 * cuda_device,
    },
}
LOCAL_SESSION_CONFIG = extend_config(SESSION_CONFIG_EXT, LOCAL_SESSION_CONFIG)
