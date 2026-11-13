import sys, threading, queue, serial, time
from PID import PIDAngle, PIDParams

PORT = "/dev/ttyACM0" #ttyACM0 # ttyUSB0
BAUD = 115200
FRAME_TIMEOUT_S = 0.05   #if for 50 ms nothing arrived discard this frame
MAX_PAYLOAD_LEN = 64
READ_TIMEOUT_S = 0.002

# cascade mapping settings (minimal P cascade)
CASCADE_MAX_X = 6000.0       # expected max |x|
CASCADE_MAX_THETA = 0.005     # max theta_ref [rad] assigned from position


# init usb communication
ser = serial.Serial(PORT, BAUD, timeout=READ_TIMEOUT_S)
time.sleep(2)                 
ser.reset_input_buffer()  

# init the que
state_que = queue.Queue() #FIFO-QUE

# init test variables
broken_frame_counter = 0
u_counter = 100.0

# create PID instance
params = PIDParams(Kp=5000.0, Ki=4000, Kd=0.0, u_min=-150.0, u_max=150.0)
pid = PIDAngle(params, dt=0.01)
theta_ref = 0.0



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
    
    ser.write(f"<{u:.1f}>\n".encode('ascii'))
    
            


def main():
    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()
    
    # open log file and counter (minimal change)
    log_file = open("log.csv", "w")
    log_file.write("count,t,u,theta,thetadot,i_term\n")
    counter = 0
    t_start = time.monotonic()   # reference for the logged time axis [s]

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
            # compute a simple P-cascade: map cart position x -> small desired angle theta_ref
            # positive x -> small negative theta_ref to steer cart back toward 0
            theta_ref = - (x / CASCADE_MAX_X) * CASCADE_MAX_THETA
            # clip
            if theta_ref > CASCADE_MAX_THETA:
                theta_ref = CASCADE_MAX_THETA
            elif theta_ref < -CASCADE_MAX_THETA:
                theta_ref = -CASCADE_MAX_THETA

            # feed shifted angle to the angle PID so it tries to reach theta_ref
            u = int(pid.step((theta, thetadot), theta_ref))

            # integral contribution to u (Ki * raw integral state) for logging
            i_term = pid.p.Ki * pid.I

            if abs(theta) > 0.4:
                send_control(0.0)
            else:
            #and send it to the arduino
                send_control(u)
            
            # write to CSV instead of printing
            #idx = next(debug_id)
            counter += 1
            t = time.monotonic() - t_start
            log_file.write(f"{counter},{t:.4f},{u:.1f},{theta:.5f},{thetadot:.5f},{i_term:.3f}\n")
            log_file.flush()
            
            
    except KeyboardInterrupt:
        # set the motor to zero
        u = 0.0
        send_control(u)
        pass
    
    finally:
        try:
            log_file.close()
        except Exception:
            pass
        ser.close()







if __name__ == "__main__":
    main()