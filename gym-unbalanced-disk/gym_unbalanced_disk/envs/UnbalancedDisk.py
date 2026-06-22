import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from os import path
import time

class UnbalancedDisk(gym.Env):
    '''
    UnbalancedDisk
    th =            
                  +-pi
                    |
           pi/2   ----- -pi/2
                    |
                    0  = starting location
    '''
    # Declares which render modes the env supports. Without this, gymnasium's
    # passive env checker (used by gym.make) rejects render_mode='human'.
    metadata = {"render_modes": ["human"], "render_fps": 40}

    def __init__(self, umax=20., dt = 0.025, max_steps=200, base_reward=True, robust=False, is_evaluation=False, reference=False, ref_angle= 15, render_mode='human'):
        """
        base_reward: If True, use the original reward function (cosine bowl + sharp Gaussian peak at the top). If False, use the new reward function with a tunable sharp peak at the target angle (Described in PPO/A2C: Reference Tracking).
        robust: If True, add noise to the action and randomly drop commands to encourage robustness (Described in PPO/A2C: model definition).
        is_evaluation: If True, use the evaluation reward function (same cosine shape as training, peaked at the target angle) instead of the training reward function (Described in PPO/A2C: Evaluation).
        reference: If True, the reward peak is ref_angle away from the top; if False, the reward peak is at the top (Described in PPO/A2C: Reference Tracking).
        ref_angle: The angle ( in degrees) away from the top where the reward peak is located if reference is True (Described in PPO/A2C: Reference Tracking).
        """
        
        ############# start do not edit  ################
        self.omega0 = 11.339846957335382
        self.delta_th = 0
        self.gamma = 1.3328339309394384
        self.Ku = 28.136158407237073
        self.Fc = 6.062729509386865
        self.coulomb_omega = 0.001

        # self.g = 9.80155078791343
        # self.J = 0.000244210523960356
        # self.Km = 10.5081817407479
        # self.I = 0.0410772235841364
        # self.M = 0.0761844495320390
        # self.tau = 0.397973147009910
        ############# end do not edit ###################

        self.umax = umax
        self.dt = dt #time step
 

        # change anything here (compilable with the exercise instructions)
        self.action_space = spaces.Box(low=-umax,high=umax,shape=tuple()) #continuous
        # self.action_space = spaces.Discrete(5) #discrete
        low = [-float('inf'),-40] 
        high = [float('inf'),40]
        
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(2,))

        # Dense reward based on disk height: 0 at the bottom (theta = 0),
        # 0.5 at horizontal, 1.0 at upright (theta = pi). Unlike a narrow
        # Gaussian, this has a non-zero gradient at every angle, so the agent
        # always has a signal telling it which way is "up" - essential for the
        # agent to discover the swing-up.
        
        self.render_mode = render_mode
        self.viewer = None
        self.u = 0 #for visual
        
        ### ADDED:Configurable reward shaping parameters (compilable with the exercise instructions) ###
        self.max_steps = max_steps
        self.reached_top = False
        self.robust = robust
        self.is_evaluation = is_evaluation
        self.max_eval_reward = max_steps
        self.ref_angle = np.deg2rad(ref_angle)
        self.reference = reference
        self.base_reward = base_reward
        ### END ADDED ###
        self.reset()

    ### ADDED: Reward function with a tunable sharp peak at the target angle (Described in PPO/A2C: Reference Tracking) ###
    def _target_angle(self):
        """Angle of the reward peak: ref_angle away from the top if set, otherwise the top."""
        return np.pi + self.ref_angle if self.reference else np.pi

    ### ADDED: Reward function with a tunable sharp peak at the target angle (Described in PPO/A2C) ###
    def reward_fun(self):
        """The reward function that evaluates how long and well the agent can keep the disk upright with a sharp peak at the target angle."""
        target = self._target_angle()
        # angle distance to the target, wrapped to [-pi, pi]
        d = np.arctan2(np.sin(self.th - target), np.cos(self.th - target))
        upright = (1 + np.cos(d)) / 2             # cosine bowl, peak 1.0 at the target

        if self.base_reward:  
            bonus = 0
        else:
            bonus = np.exp(-(d / 0.25)**2)            # sharp, ~14° width, peak 1.0
            

        return upright + 0.5 * bonus

    ### ADDED: A separate evaluation reward function (Used for PPO/A2C Evaluation) ###
    def evaluation_reward_fun(self):
        """The evaluation reward function: same cosine shape as training, peaked at the target angle."""
        target = self._target_angle()
        d = np.arctan2(np.sin(self.th - target), np.cos(self.th - target))
        return (1 + np.cos(d)) / 2


    def step(self, action):
        #convert action to u
    
        self.u = action 


        ### ADDED: Robustness features (Described in PPO/A2C: model definition) ###
        if self.robust:
            self.u += np.random.normal(loc=0, scale=0.5)   # actuator amplitude noise
            if np.random.rand() < 0.05:                     # 5% dropped command:
                self.u = self.u_prev                        # amplifier holds last torque
        self.u_prev = self.u

        ##### Start Do not edit ######
        self.u = np.clip(self.u,-self.umax,self.umax)
        def f(t,y):
            th, omega = y
            dthdt = omega
            friction = self.gamma*omega + self.Fc*np.tanh(omega/self.coulomb_omega)
            domegadt = -self.omega0**2*np.sin(th+self.delta_th) - friction + self.Ku*self.u
            return np.array([dthdt, domegadt])
        sol = solve_ivp(f,[0,self.dt],[self.th,self.omega]) #integration
        self.th, self.omega = sol.y[:,-1]
        ##### End do not edit   #####


        ### Flag for evalutaion reward (Described in PPO/A2C: Evaluation) ###
        if self.is_evaluation:
            reward = self.evaluation_reward_fun()
        else:
            reward = self.reward_fun()

        # Check if episode is done
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        return self.get_obs(), reward, False, truncated, {}
         
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.th = self.np_random.normal(loc=0, scale=0.001)
        self.omega = self.np_random.normal(loc=0, scale=0.001)
        # Reset step counter and action for rendering
        self.current_step = 0
        self.reached_top = False
        self.u = 0
        self.u_prev = 0
        return self.get_obs(), {}

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        # Match the float32 observation_space (avoids gymnasium dtype / "not in
        # observation space" warnings caused by the default float64 array).
        return np.array([self.th_noise, self.omega_noise], dtype=np.float32)

    def render(self):
        import pygame
        from pygame import gfxdraw
        
        screen_width = 500
        screen_height = 500

        th = self.th
        omega = self.omega #x = self.state

        if self.viewer is None:
            pygame.init()
            pygame.display.init()
            self.viewer = pygame.display.set_mode((screen_width, screen_height))

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        gfxdraw.filled_circle( #central blue disk
        
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.65*1.3),
            (32,60,92),
        )
        gfxdraw.filled_circle( #small midle disk
            self.surf,
            screen_width//2,
            screen_height//2,
            int(screen_width/2*0.06*1.3),
            (132,132,126),
        )
        
        from math import cos, sin
        r = screen_width//2*0.40*1.3
        gfxdraw.filled_circle( #disk
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22*1.3),
            (155,140,108),
        )
        gfxdraw.filled_circle( #small nut
            self.surf,
            int(screen_width//2-sin(th)*r), #is direction correct?
            int(screen_height//2-cos(th)*r),
            int(screen_width/2*0.22/8*1.3),
            (71,63,48),
        )
        
        fname = path.join(path.dirname(__file__), "clockwise.png")
        self.arrow = pygame.image.load(fname)
        if self.u:
            if isinstance(self.u, (np.ndarray,list)):
                if self.u.ndim==1:
                    u = self.u[0]
                elif self.u.ndim==0:
                    u = self.u
                else:
                    raise ValueError(f'u={u} is not the correct shape')
            else:
                u = self.u
            arrow_size = abs(float(u)/self.umax*screen_height)*0.25
            Z = (arrow_size, arrow_size)
            arrow_rot = pygame.transform.scale(self.arrow,Z)
            if self.u<0:
                arrow_rot = pygame.transform.flip(arrow_rot, True, False)
                
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.viewer.blit(self.surf, (0, 0))
        if self.u:
            self.viewer.blit(arrow_rot, (screen_width//2-arrow_size//2, screen_height//2-arrow_size//2))
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()

        return True

    def close(self):
        if self.viewer is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.viewer = None


class UnbalancedDisk_sincos(UnbalancedDisk):
    """docstring for UnbalancedDisk_sincos"""
    def __init__(self, umax=3., dt = 0.025, base_reward=False, is_evaluation=False,  max_steps=200, robust=False, reference=False, ref_angle=15, render_mode='human'):
        super(UnbalancedDisk_sincos, self).__init__(umax=umax, dt=dt, base_reward=base_reward, is_evaluation=is_evaluation, max_steps=max_steps, robust=robust, reference=reference, ref_angle=ref_angle, render_mode=render_mode)
        low = [-1,-1,-40.] 
        high = [1,1,40.]
        self.observation_space = spaces.Box(low=np.array(low,dtype=np.float32),high=np.array(high,dtype=np.float32),shape=(3,))

    def get_obs(self):
        self.th_noise = self.th + np.random.normal(loc=0,scale=0.001) #do not edit
        self.omega_noise = self.omega + np.random.normal(loc=0,scale=0.001) #do not edit
        return np.array([np.sin(self.th_noise), np.cos(self.th_noise), self.omega_noise]) #change anything here

if __name__ == '__main__':
    import time
    env = UnbalancedDisk(dt=0.025)

    obs = env.reset()
    Y = [obs]
    env.render()
    try:
        act = [3, -3]
        now = time.time()
        for i in range(100):
            time.sleep(1/24)
            # switch every 2 seconds between 3 and -3, otherwise do not apply any action
            u = act[i//(2*24)%2]
            obs, reward, terminated, truncated, info = env.step(u)
            Y.append(obs)
            env.render()
    finally:
        env.close()
    from matplotlib import pyplot as plt
    import numpy as np
    Y = np.array(Y)
    plt.plot(Y[:,0])
    plt.title(f'max(Y[:,0])={max(Y[:,0])}')
    plt.show()
    

