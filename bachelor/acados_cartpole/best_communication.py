import sys, threading, queue, serial, time, itertools

PORT = "/dev/ttyACM0"
BAUD = 115200
FRAME_TIMEOUT_S = 0.05   #if for 50 ms nothing arrived discard this frame
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

# init usb communication
ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)                 
ser.reset_input_buffer()  

#init the que
state_que = queue.Queue() #FIFO-QUE

# init test variables
broken_frame_counter = 0
u_counter = 100.0
debug_id = itertools.count(1)



def listen_to_arduino():
    """
    -reads frames in the background with threading
    -frames come in the form  of <x,v,theta,thetadot>\n
    -if full frame got received -> put it in a que as tuple
    """
    
    currently_receiving = False
    buffer = bytearray()
    t0 = None
    global broken_frame_counter
    
    while True:
        b = ser.read(1)
        if not b:
            # probably something wrong with the frame
            if currently_receiving and (time.monotonic() - t0 > FRAME_TIMEOUT_S):
                currently_receiving = False
                buffer.clear()
                t0 = None
                broken_frame_counter += 1
            continue 
        
        c = b[0]
        
        # we are at the beginning of the payload
        if currently_receiving == False: 
            if c == ord('<'):
                currently_receiving = True
                buffer.clear()
                t0 = time.monotonic()
            continue
        
        # we are at the end of the payload
        if c == ord('>'): 
            try:
                payload = buffer.decode('ascii', errors='strict').strip()
                # handshake only once this branch is taken
                if payload.lower() == "ready":
                    state_que.put("ready")
                else:
                    parts = payload.split(',')
                    if len(parts) == 4:
                        x, v, theta, thetadot = map(float, parts)
                        state_que.put((x, v, theta, thetadot))
                    else:
                        broken_frame_counter += 1
                        
            except Exception:
                broken_frame_counter += 1
                    
            currently_receiving = False
            buffer.clear()
            t0 = None
            continue
            
        
        elif c in (ord('\n'), ord('\r')):  
            continue
        
        # we are inside the payload and buffer is smaller than max allowed length
        if len(buffer) < MAX_PAYLOAD_LEN:
            buffer.append(c)
        
        # overshoot of max length -> discard the reading
        else:
            currently_receiving = False
            buffer.clear()
            t0 = None
            broken_frame_counter += 1
            

def send_control(u: float):
    """
    -sends <±xx.xx> + \n 
    -newline only for debugging
    """
    
    ser.write(f"<{u:.3f}>\n".encode('ascii'))
    
    
    
def test_fct(x: float, v: float, theta: float, thetadot: float):
    global u_counter
    u_counter += 0.001
    return u_counter
            


def main():
    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()
    
    # while True:
    #     print("waiting for Arduino Handshake")
    #     print(broken_frame_counter)
    #     msg = state_que.get() # blocks and waits for the arduino to end the setup
    #     if msg == "ready":
    #         print("Arduino Ready")
    #         break
        
    
    try:
        while True:
            # wait for a new state
            state = state_que.get() # blocks untill the thread adds an item
            
            while True:
                try: 
                    state = state_que.get_nowait()
                
                except queue.Empty:
                    break
            
            # just in case something else is in the que
            if not (isinstance(state, tuple) and len(state) == 4):
                continue
            
            # we finally got the newest state and extract it now
            x, v, theta, thetadot = state 

            # now apply the control application
            u = test_fct(x,v, theta, thetadot)
            
            # and send it to the arduino
            send_control(u)
            
            # for debugging
            idx = next(debug_id)
            print(f"{x:.3f},{v:.3f},{theta:.3f},{thetadot:.3f},{idx},{broken_frame_counter}")
            
            
    except KeyboardInterrupt:
        pass
    
    finally:
        ser.close()







if __name__ == "__main__":
    main()