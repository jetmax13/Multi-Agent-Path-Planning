import numpy as np

def create_obstacles(sim_time, num_timesteps):
    # Obstacle 1
    v = -2
    p0 = np.array([5, 12])
    obst = create_robot(p0, v, np.pi/2, sim_time,
                        num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = obst
    # Obstacle 2
    v = 2
    p0 = np.array([0, 5])
    obst = create_robot(p0, v, 0, sim_time, num_timesteps).reshape(
        4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    # Obstacle 3
    v = 2
    p0 = np.array([10, 10])
    obst = create_robot(p0, v, -np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
                                                                                num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))
    # Obstacle 4
    v = 2
    p0 = np.array([7.5, 2.5])
    obst = create_robot(p0, v, np.pi * 3 / 4, sim_time, num_timesteps).reshape(4,
                                                                               num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    # Obstacle 5
    v = 2
    p0 = np.array([10.0, 2.0])
    obst = create_robot(p0, v, np.pi / 2, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    # Obstacle 6
    v = 1.8
    p0 = np.array([2.0, 10.0])
    obst = create_robot(p0, v, np.pi, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    # Obstacle 7
    v = 2.2
    p0 = np.array([6.0, 6.0])
    obst = create_robot(p0, v, np.pi / 3, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    # Obstacle 8
    v = 1.5
    p0 = np.array([8.0, 8.0])
    obst = create_robot(p0, v, -np.pi / 2, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    # Obstacle 9
    v = 2.1
    p0 = np.array([3.0, 3.0])
    obst = create_robot(p0, v, -np.pi / 4, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))

    # Obstacle 10
    v = 1.7
    p0 = np.array([9.0, 5.0])
    obst = create_robot(p0, v, np.pi, sim_time, num_timesteps).reshape(4, num_timesteps, 1)
    obstacles = np.dstack((obstacles, obst))


    return obstacles


def create_robot(p0, v, theta, sim_time, num_timesteps):
    # Creates obstacles starting at p0 and moving at v in theta direction
    t = np.linspace(0, sim_time, num_timesteps)
    theta = theta * np.ones(np.shape(t))
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    v = np.stack([vx, vy])
    p0 = p0.reshape((2, 1))
    p = p0 + np.cumsum(v, axis=1) * (sim_time / num_timesteps)
    p = np.concatenate((p, v))
    return p
