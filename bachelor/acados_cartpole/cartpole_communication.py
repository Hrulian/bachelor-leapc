import threading, queue, serial, time

from my_helpers import force_to_pwm, state_tuple_to_tensor
from my_planner import CartPolePlannerConfig, CartPolePlanner, create_custom_cartpole_params


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

# init the acados controller
cfg = CartPolePlannerConfig()
params = create_custom_cartpole_params("stagewise", cfg.N_horizon)
controller = CartPolePlanner(cfg, params)
ctx = None



def listen_to_arduino():
    """
    -reads frames in the background with threading
    -frames come in the form  of <x,theta,v,thetadot>\n
    -if full frame got received -> put it inn a que as tuple
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
    

def main():
    # ensures we reference the module-level variables
    global ctx  
    
    # start the background receiver task
    arduinoThread = threading.Thread(target=listen_to_arduino, args=())
    arduinoThread.daemon = True
    arduinoThread.start()
        
    
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
            x, theta, v, thetadot = state 

            # prepare state as torch tensor for the planner
            state_converted = state_tuple_to_tensor(state)
            
            # call planner: returns (ctx, u0, x_traj, u_traj, value)
            ctx, u0, x_traj, u_traj, value = controller(state_converted, ctx=ctx)
            
            # extract first control
            u_force = float(u0.detach().cpu().numpy().squeeze().item())

            # convert first from N to PWM
            u = force_to_pwm(u_force, v, 50)

            # send PWM control to arduino
            send_control(u)
            
            # for debugging
            print(f"{x},{theta:.3f},{v},{thetadot:.3f}, -> {u_force:.3f}N -> PWM {u}")
            
    except KeyboardInterrupt:
        send_control(0.0)
        pass
    
    finally:
        ser.close()




if __name__ == "__main__":
    main()