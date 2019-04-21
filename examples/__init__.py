def default_config_path(agent_name, env_name):
    return 'configs/{}_{}.json'.format(agent_name, env_name)


def default_save_folder(agent_name, env_name):
    return '{}_{}_save_point'.format(agent_name, env_name)


__all__ = ['default_config_path', 'default_save_folder']
