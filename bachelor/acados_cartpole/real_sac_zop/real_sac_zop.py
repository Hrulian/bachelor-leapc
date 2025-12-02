

"""
Plan structure for real_sac_zop.py_

-policy in parameter space 
-parameter critic + target copy
-ReplayBuffer (buffer.py)
-MPC like before with acados but the planner wrapped into controller class and then into actor class (MpcSacActor)

    ->try to extract those from existing sac_zop files
    
---------------------------------------------------------------

Main loop:
    1. read state from Arduino, calculate reward + done and put into buffer (state, param, reward, next_state, done)
    2. call MpcSacActor with current state it then internally uses the MPC to get the force to be applied
    3. send force to Arduino
    4. call sac_zop update step every N steps
    5. repeat




"""

