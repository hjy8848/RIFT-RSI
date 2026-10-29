"""Atomic journals, external profile pin, no silent provider retry or selective rates."""
import fcntl
import json
import os
from pathlib import Path
import tempfile
from .benchmark import Run, load, validate, run_instance, summarize


def atomic_json(path,value):
    if hasattr(value,'model_dump'):value=value.model_dump(mode='json')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as file:
            json.dump(value,file,ensure_ascii=False,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)


def run_suite(m,out,clients,max_runs=None):
    validate(m);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        profile=out/'profile.json'
        if profile.exists():
            if load(profile)!=m:raise ValueError('existing suite directory has a different frozen profile')
        else:atomic_json(profile,m)
        runs=[];calls=0
        for i in m.instances:
            for trial in m.trials:
                name=i.id+'.'+trial
                path=out/(name+'.run.json');started=out/(name+'.started.json')
                if path.exists():
                    runs.append(Run.model_validate_json(path.read_text(encoding='utf-8')));continue
                if started.exists():raise ValueError('incomplete prior attempt; no silent retry. Preregister a fresh trial/profile in a new directory')
                if max_runs is not None and calls>=max_runs:continue
                atomic_json(started,{'manifest_hash':m.manifest_hash,'instance_id':i.id,'trial':trial})
                service,customer=clients(m,i)
                try:run=run_instance(m,i,service,customer,trial)
                finally:
                    for client in (service,customer):
                        if client is not None and hasattr(client,'close'):client.close()
                atomic_json(path,run);runs.append(run);calls+=1
                print(f'{i.id}/{trial}: {(run.execution or run.interaction.service_execution).status}',flush=True)
        report=summarize(m,runs);atomic_json(out/'summary.json',report);return report
