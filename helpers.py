import time
import csv

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(list(map(int, seq[int(last):int(last + avg)])))
        last += avg

    return out


def save_to_csv(file_name, data, n):
  logs = open("logs_{0}_{1}.csv".format(file_name, n, time.strftime("%Y%m%d")), 'a')
  with logs:
     writer = csv.writer(logs)
     writer.writerows(data)
