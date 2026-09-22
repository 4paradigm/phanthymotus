"""Offline four-stream correlation. No robot imports, commands or accuracy gate."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import json
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


def gaps(stamps):
    ordered = sorted(set(stamps))
    return max((b-a for a,b in zip(ordered,ordered[1:])), default=0)/1e6


def distribution(values):
    values = sorted(values)
    return {'count':len(values), 'mean':sum(values)/len(values) if values else None,
            'p95':values[min(len(values)-1,int(len(values)*.95))] if values else None,
            'max':values[-1] if values else None}


def motor_q(row):
    values={m['id']:m['q_rad'] for m in row.get('motors',[])}
    ids=list(range(11,18))+list(range(21,28))
    return [values[i] for i in ids] if all(i in values for i in ids) else None


def correlate(vendor, commands, window_ms=100, encoding_tolerance=1e-6):
    """Candidate matches, not invented sequence IDs. Repeated holds can be ambiguous."""
    commands=sorted(commands,key=lambda r:r['observed_ns'])
    stamps=[r['observed_ns'] for r in commands];out=[]
    for event in vendor:
        start=event['publish_started_ns'];end=event['publish_returned_ns']+int(window_ms*1e6)
        candidates=[]
        for index in range(bisect_left(stamps,start),bisect_right(stamps,end)):
            actual=motor_q(commands[index]);goal=event['q_rad']
            if actual is not None and max(abs(a-b) for a,b in zip(actual,goal))<=encoding_tolerance:
                candidates.append({'observer_index':index,'observed_ns':stamps[index],
                                   'receipt_after_publish_start_ms':(stamps[index]-start)/1e6})
        out.append({'event_id':event['event_id'],'session_id':event.get('session_id'),
                    'sequence':event.get('seq'),'candidate_count':len(candidates),
                    'candidates':candidates,'certainty':'unique_candidate' if len(candidates)==1
                    else 'ambiguous' if candidates else 'unobserved_not_proof_of_no_publication'})
    return out


def summarize(directory):
    body=rows(directory/'ros-body.jsonl');feedback=rows(directory/'ros-driver.jsonl')
    events={};declared={};ready=[];end=[];other_traces=set()
    start_path=directory/'trace-start.json'
    active_trace=json.loads(start_path.read_text()).get('trace_id') if start_path.exists() else None
    final_trace=json.loads((directory/'trace-stop.json').read_text()) if (directory/'trace-stop.json').exists() else None
    for row in body+feedback+([{'trace':final_trace}] if final_trace else []):
        if row.get('event')=='observer_ready':ready.append(row)
        if row.get('event')=='observer_end':end.append(row)
        trace=row.get('trace') or {}
        key=trace.get('trace_id')
        # Discovery precedes trace_start and can receive the previous run's ring.
        # Keep those raw rows, but never count their history as this run's loss.
        if key and active_trace and key!=active_trace:
            other_traces.add(key)
            continue
        if key:
            declared[key]=max(declared.get(key,0),trace['event_count'])
            for event in trace['events']:events[(key,event['event_id'])]=event
    driver=sorted(events.values(),key=lambda e:e['monotonic_ns'])
    missing={key:[i for i in range(1,count+1) if (key,i) not in events] for key,count in declared.items()}
    commands=[r for r in body if r.get('event')=='arm_cmd_pos']
    status=[r for r in body if r.get('event')=='arm_status']
    apply=rows(directory/'actucore.jsonl');independent=rows(directory/'round-1.jsonl')
    combined=rows(directory/'combined.jsonl');stages={}
    for name,source,timekey in [('execution',independent,'monotonic_ns'),('full_chain',combined,'observed_ns')]:
        stamps=[r[timekey] for r in source if r.get(timekey)]
        if not stamps:
            stages[name]={'available':False};continue
        low,high=min(stamps),max(stamps)
        selected=[e for e in driver if low<=e['monotonic_ns']<=high]
        decisions=[e for e in selected if e['event']=='command_decision']
        vendor=[e for e in selected if e['event']=='vendor_publish']
        observed=[r for r in commands if low<=r['observed_ns']<=high]
        measured=[r for r in status if low<=r['observed_ns']<=high]
        source_apply=[r for r in apply if low<=r['monotonic_ns']<=high]
        publications=([r['publish'] for r in source if r.get('publish')] if name=='execution'
                      else [r['publish'] for r in source_apply if r.get('publish')])
        failures=Counter((r.get('failure') or {}).get('code') for r in source_apply if r.get('failure'))
        recoveries=[];waiting=None
        for r in source_apply:
            if r.get('failure') and waiting is None:waiting=r
            if r.get('published') and waiting is not None:
                recoveries.append({'failure_input_sequence':waiting['input_sequence'],
                    'resumed_input_sequence':r['input_sequence'],
                    'reason':waiting['failure']['code'],
                    'elapsed_ms':(r['monotonic_ns']-waiting['monotonic_ns'])/1e6,
                    'same_session':r.get('session_id')==waiting.get('session_id')})
                waiting=None
        errors=[];reference_errors=[]
        for r in source:
            q=r.get('q');goal=r.get('sent_target_q') or r.get('target_q');reference=r.get('ik_reference_q')
            if q and goal:errors.extend(abs(a-b) for a,b in zip(q,goal))
            if q and reference:reference_errors.extend(abs(a-b) for a,b in zip(q,reference))
        matches=correlate(vendor,commands)
        stages[name]={'available':True,'window_ns':[low,high],'input_rows':len(source),
            'replay_inputs_superseded':sum(r.get('observer_timing_ms',{}).get('superseded_count',0) for r in source),
            'actucore_apply_count':len(source_apply),'ik_success_count':sum(r['ik_succeeded'] for r in source_apply),
            'apply_failure_reasons':dict(failures),
            'ik_failure_reasons':{k:v for k,v in failures.items() if k and k.startswith('ik_')},
            'actucore_publication_count':len(publications),
            'driver_selected_received':len(decisions),'driver_accepted':sum(e['accepted'] for e in decisions),
            'driver_rejected_reasons':dict(Counter(e.get('reason') for e in decisions if not e['accepted'])),
            'socket_superseded':sum(e.get('superseded',0) for e in selected),
            'vendor_publications':dict(Counter(e['kind'] for e in vendor)),
            'ros_cmd_receipts':len(observed),'ros_status_receipts':len(measured),
            'ros_error_codes':dict(Counter(str(m['error']) for r in measured for m in r['motors'] if m['error'])),
            'max_abs_measured_velocity_rad_s':max((abs(m['dq_rad_s']) for r in measured for m in r['motors']),default=None),
            'longest_publish_gap_ms':gaps(p['published_ns'] for p in publications),
            'longest_vendor_gap_ms':gaps(e['publish_started_ns'] for e in vendor),
            'longest_status_gap_ms':gaps(r['observed_ns'] for r in measured),
            'recoveries':recoveries,'unrecovered_failure':waiting,
            'command_error_rad':distribution(errors),'full_ik_error_rad':distribution(reference_errors),
            'vendor_ros_matching':dict(Counter(m['certainty'] for m in matches))}
        (directory/(name+'-vendor-matches.json')).write_text(json.dumps(matches,indent=2)+'\n')
    boot_ids={r['boot_id'] for r in ready}
    result={'tracking_error_policy':'report_only','same_robot_monotonic_clock':len(boot_ids)==1 and len(ready)==2,
            'observer_ends':len(end),'observer_queue_drops':sum(r['queue_dropped'] for r in end),
            'driver_missing_event_ids':missing,'driver_trace_id':active_trace,
            'other_trace_ids_observed':sorted(other_traces),'stages':stages,
            'management_rejections':dict(Counter(e.get('reason') for e in driver
                if e['event']=='management_result' and not e['accepted'])),
            'ros_graph_snapshots':[r for r in body+feedback if r.get('event')=='ros_graph'],
            'power_feedback_receipt_age_ms':distribution((r['observed_ns']-r['feedback']['power_ns'])/1e6
                for r in feedback if (r.get('feedback') or {}).get('power_ns')),
            'power_topic_receipts':sum(r.get('event')=='power_status' for r in body),
            'power_topic_longest_gap_ms':gaps(r['observed_ns'] for r in body if r.get('event')=='power_status'),
            'limitations':['DDS best-effort receipts do not prove zero loss.',
                'Vendor messages have no sequence; matches use a 100 ms receipt window and 1e-6 rad encoding tolerance, not a tracking-error gate.',
                'Window bounds use available source evidence; missing tail or unresolved failures are not declared successful.',
                'Superseded latest-target frames are allowed; input and execution counts need not agree.']}
    (directory/'chain-report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(summarize(parser.parse_args().directory),indent=2))
