"""Bot build-plan checks live with bot dependencies, not runtime image tests."""

def test_existing_bot_command_selects_unified_jp61_build(tmp_path, monkeypatch):
    import asyncio
    from agents.pr_review import builder, worker
    from agents.pr_review.config import Config
    from agents.pr_review.models import BuildTarget, ReviewJob, parse_trigger_command

    parsed = parse_trigger_command('/request_bot_review core actucore jp61')
    job = ReviewJob(repo_full_name='example/repo', pr_number=1, pr_head_sha='abc1234',
        pr_head_ref='test', pr_base_ref='main', comment_id=1, requester='tester',
        force_targets=parsed['force_targets'], perception_variants=parsed['perception_variants'])
    targets, paths = worker._parse_forced_targets(job.force_targets)
    assert worker._build_plan(job, targets, paths) == [
        (BuildTarget.CORE, None, ''), (BuildTarget.ACTUCORE, None, '6.1')]
    recorded = []
    async def sink(**kwargs): recorded.append(kwargs)
    monkeypatch.setattr(builder, '_build_with_script', sink)
    asyncio.run(builder.build_actucore(tmp_path, Config(), tmp_path / 'log', '6.1'))
    assert recorded[0]['args'] == ['--mirror', 'tencent', '--jp-version', '6.1']
    assert recorded[0]['script'] == tmp_path / 'deploy/build_actucore.sh'
