def getAllTimestamps(data):
    timestamps = []
    for row in data:
        row = row.decode('utf-8').split('::')
        timestamp = row[3]
        timestamps.append(int(timestamp))
    return timestamps

if __name__ == '__main__':
    data_file_path = '/home/FYP/siddhant005/fyp/code/data/raw/ml-1m/ratings.dat'
    data = open(data_file_path, 'rb').readlines()
    ts = getAllTimestamps(data)
    print(len(ts))
    print(min(ts))