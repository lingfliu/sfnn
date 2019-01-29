from multiprocessing import Pool, Queue, Process, Manager
import time


def task(i, queue):
    # print('runs here', queue)
    queue.put((i, i+1, i+2))


if __name__ == '__main__':
    pool = Pool(processes=10)
    queue = Manager().Queue(maxsize=500)

    procs = []
    res = []
    for i in range(50):
        # proc = Process(target=task, args=(i, queue, ))
        pool.apply_async(task, args=(i, queue, ))
        # res.append(pool.apply_async(task, args=(i, queue, )))

        # procs.append(proc)
        # proc.start()
    pool.close()

    while True:
        i = queue.get()
        print('queue pull = ', i)

    pool.join()
    # for proc in procs:
    #     proc.join()
    # pool.close()
    # pool.join()




