from athena import gpu_ops as ad

import os
import sys
import yaml
import multiprocessing
import signal


def main():
    def start_scheduler(settings):
        for key, value in settings.items():
            os.environ[key] = str(value)
        assert os.environ['DMLC_ROLE'] == "scheduler"
        print('Scheduler starts...')
        ad.scheduler_init()
        ad.scheduler_finish()
    def start_server(settings):
        for key, value in settings.items():
            os.environ[key] = str(value)
        assert os.environ['DMLC_ROLE'] == "server"
        print('Server starts...')
        ad.server_init()
        ad.server_finish()
    def signal_handler(sig, frame):
        print("SIGINT signal caught, stop Training")
        for proc in server_procs:
            proc.kill()
        sched_proc.kill()
        exit(0)

    if len(sys.argv) == 1:
        settings = yaml.load(open('./settings/dist_s1.yml').read(), Loader=yaml.FullLoader)
    else:
        file_path = sys.argv[1]
        suffix = file_path.split('.')[-1]
        if suffix == 'yml':
            settings = yaml.load(open(file_path).read(), Loader=yaml.FullLoader)
        else:
            assert False, 'File type not supported.'    
    print('Scheduler and servers settings:')
    print(settings)

    server_procs = []
    for key, value in settings.items():
        if key == 'shared':
            continue
        elif key == 'sched':
            sched_proc = multiprocessing.Process(target=start_scheduler, args=(value,))
            sched_proc.start()
        else:
            server_procs.append(multiprocessing.Process(target=start_server, args=(value,)))
            server_procs[-1].start()

    signal.signal(signal.SIGINT, signal_handler)
    for proc in server_procs:
        proc.join()
    sched_proc.join()


if __name__ == '__main__':
    main()
