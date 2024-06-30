import threading
import time

def thread_function(thread_number, start_event, stop_event):
    while not stop_event.is_set():
        if start_event.is_set():
            start_time = time.time()  # Reset start time when the thread is (re)started
            while start_event.is_set() and not stop_event.is_set():
                elapsed_time = int(time.time() - start_time)
                print(f"Thread {thread_number} is running at {elapsed_time}")
                time.sleep(5)

# Events to control the threads
start_event_1 = threading.Event()
start_event_2 = threading.Event()
start_event_3 = threading.Event()
stop_event_1 = threading.Event()
stop_event_2 = threading.Event()
stop_event_3 = threading.Event()

# Creating threads
thread1 = threading.Thread(target=thread_function, args=(1, start_event_1, stop_event_1))
thread2 = threading.Thread(target=thread_function, args=(2, start_event_2, stop_event_2))
thread3 = threading.Thread(target=thread_function, args=(3, start_event_3, stop_event_3))

# Start threads 1 and 3
start_event_1.set()
start_event_3.set()
thread1.start()
thread3.start()

# Wait for 20 seconds
time.sleep(20)

# Stop thread 1 and start thread 2
stop_event_1.set()
start_event_2.set()
thread2.start()

# Wait for 18 seconds
time.sleep(18)

# Stop thread 3 and start thread 1 again
stop_event_3.set()
start_event_1.clear()  # Clear the start event for thread 1
stop_event_1.clear()
start_event_1.set()

# Ensure all threads are joined before exiting
thread1.join()
thread2.join()
thread3.join()