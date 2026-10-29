"""Atomic preregistered attempts. A started failed attempt is never retried."""
import fcntl
import json
import os
import tempfile
from pathlib import Path
from .benchmark import Run, Manifest, validate, run_instance, summarize


def atomic_json(path,value):
    if hasattr(value,'model_dump'):
        value=value.model_dump(mode='json')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as file:
            json.dump(value,file,ensure_ascii=False,indent=2);file.write('\n');file.flush();os.fsync(file.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def run_suite(adapter,manifest,out,clients,max_runs=None):
    validate(adapter,manifest)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        profile=out/'profile.json'
        if profile.exists():
            if Manifest.model_validate_json(profile.read_text())!=manifest:
                raise ValueError('existing suite has another frozen profile')
        else:
            atomic_json(profile,manifest)
        runs=[];attempts=0
        for instance in manifest.instances:
            for trial in manifest.trials:
                name=instance.id+'.'+trial
                path,started=out/(name+'.run.json'),out/(name+'.started.json')
                if path.exists():
                    runs.append(Run.model_validate_json(path.read_text()));continue
                if started.exists():
                    raise ValueError('unfinished prior attempt; register a fresh trial/profile, no silent retry')
                if max_runs is not None and attempts>=max_runs:
                    continue
                atomic_json(started,{'manifest_hash':manifest.manifest_hash,'instance_id':instance.id,'trial':trial})
                service=customer=None
                try:
                    service,customer=clients(manifest,instance)
                    run=run_instance(adapter,manifest,instance,service,customer,trial)
                except BaseException as exc:
                    atomic_json(out/(name+'.failure.json'),{'error_type':type(exc).__name__,
                        'message':'Attempt interrupted or could not initialize; original registration retained. No automatic retry.',
                        'manifest_hash':manifest.manifest_hash})
                    raise
                finally:
                    for client in (service,customer):
                        if client is not None and hasattr(client,'close'):client.close()
                atomic_json(path,run);runs.append(run);attempts+=1
                print(f'{adapter.name} {instance.id}/{trial}: {(run.execution or run.interaction.service).status}',flush=True)
        report=summarize(adapter,manifest,runs);atomic_json(out/'summary.json',report)
        return report
