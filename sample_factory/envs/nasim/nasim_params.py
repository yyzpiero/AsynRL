def nasim_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_mujoco',
        hidden_size=64,
        encoder_extra_fc_layers=True,
        #env_frameskip=1,
        nonlinearity='silu'
    )

# noinspection PyUnusedLocal
def add_nasim_env_args(env, parser):
    p = parser
