from Agent import Agent

GAME = 'CartPole-v0'
MAX_EPISODES = 100

def main():
    A = Agent(GAME)
    A.run(MAX_EPISODES)

if __name__ == '__main__':
    main()