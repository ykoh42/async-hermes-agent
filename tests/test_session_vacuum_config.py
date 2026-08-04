def test_default_config_exposes_vacuum_interval():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["sessions"]["min_vacuum_interval_days"] == 30
