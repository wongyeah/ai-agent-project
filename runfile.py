import time
data = []
for _ in range(100):
    data.append(bytearray(50 * 1024 * 1024))
    time.sleep(0.1)
