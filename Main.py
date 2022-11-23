import pygame
from Player import Player
from Score import Score
from Pipes import Pipes
import random
import Models
import numpy as np
import tensorflow as tf
from tensorflow import keras

pygame.init()

'''
Setting up the Q-learning algorithm for Reinforcement Learning
'''

# Instantiate the Actions & Rewards
actions = ["jump", "none"]

rewards = { 
    "ground": -10, 
    "ceiling": -10, 
    "in_air": 10, 
    "through_pipe": 500
}


# Function for Loading in and Scaling Images
def convert_scale_images(urlPath, imageSize):
    curImg = pygame.image.load("images/" + urlPath).convert_alpha()
    return pygame.transform.scale(curImg, imageSize)

# Set up Initial Window
pygame.display.set_caption('Flappy Bird')
screen_width, screen_height = 1300, 700
screen = pygame.display.set_mode([screen_width, screen_height])

# Converting & Scaling images for use
flappyBirdImg = convert_scale_images("Bird.png", (150, 150))
bottomPipeImg = convert_scale_images("bottom_pipe.png", (92, 528))
topPipeImg = convert_scale_images("top_pipe.png", (92, 528))
backgroundImg = convert_scale_images("background.png", (screen_width, screen_height))

# Sets up the game clock
clock = pygame.time.Clock()

running, mainFlappyBird, numTimesContinueMovingUp, numTimesWaitNewPipe, allPipeInstances, gameOver = [None for x in range(6)]

# Generate the initial two models
Models.generate_initial_models()

def startGame():
    # Instantiate the Player
    global running, mainFlappyBird, numTimesContinueMovingUp, numTimesWaitNewPipe, allPipeInstances, gameOver
    running = True
    mainFlappyBird = Player()
    numTimesContinueMovingUp = 10
    gameOver = False
    allPipeInstances = []
    numTimesWaitNewPipe = 0
    Score.curScore = 0

for epoch in range(0, 100):
    scores = []
    print("======\n\n\n\n\nSTARTING NEW EPOCH\n\n\n\n\n======")
    models = [Models.load_models()[0], Models.load_models()[1]]
    for x in range(8):
        models.append(Models.create_random_variation(models[0], models[1]))
    for x in range(10):
        curRewards = 0
        startGame()
        # When the Pygame simulator starts running
        while running:
            # Check if the user has already lost the game
            if not gameOver:
                clock.tick(60)
                userClickedSpace = False

                # Display the current user score
                screen.blit(backgroundImg, (0, 0))
                font = pygame.font.SysFont(None, 64)
                img = font.render("Score: " + str(Score.curScore) + "\n Epoch: " + str(epoch) + "\n Run: " + str(x), True, pygame.Color(255, 255, 255))
                screen.blit(img, (30, 30))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    # if event.type == pygame.KEYDOWN:
                    #     if event.key == pygame.K_SPACE:
                    #         mainFlappyBird.move_up()
                # Create a new pipe if it is time
                if numTimesWaitNewPipe == 0:
                    allPipeInstances.append(Pipes())
                    numTimesWaitNewPipe += random.randint(100, 150)

                # Decide whether to jump up
                currentPrediction = Models.predict_for_model(models[x], mainFlappyBird.velocity, mainFlappyBird.curX, mainFlappyBird.curY, allPipeInstances[0].bottomY)
                if currentPrediction[0][0] < currentPrediction[0][1]:
                    mainFlappyBird.move_up()

                # Move the pipes on the screen, and delete the pipe if it is off the screen
                rewardThisIter = ""
                for idx, pipe in enumerate(allPipeInstances):
                    screen.blit(topPipeImg, (pipe.curX - 5, pipe.topY))
                    screen.blit(bottomPipeImg, (pipe.curX - 5, pipe.bottomY))
                    allPipeInstances[idx].curX -= 5
                    if allPipeInstances[idx].curX < -20:
                        allPipeInstances.pop(0)
                    if allPipeInstances[idx].curX == 200:
                        Score.increment_score()
                        rewardThisIter = "through_pipe"

                if rewardThisIter == "":
                    if mainFlappyBird.hit_ceiling():
                        rewardThisIter = "ceiling"
                    elif mainFlappyBird.hit_ground():
                        rewardThisIter = "ground"
                    else:
                        rewardThisIter = "in_air"

                curRewards += rewards[rewardThisIter]

                # Check if the player has crashed into the pipe
                for idx, pipe in enumerate(allPipeInstances):
                    if allPipeInstances[idx].curX - 100 <= mainFlappyBird.curX <= allPipeInstances[idx].curX + 40:
                        if mainFlappyBird.curY < allPipeInstances[idx].topY + 480 or mainFlappyBird.curY > allPipeInstances[idx].bottomY - 100:
                            gameOver = True  

                mainFlappyBird.update()
                screen.blit(flappyBirdImg, (mainFlappyBird.curX, mainFlappyBird.curY))
                numTimesWaitNewPipe -= 1

            # If the user has not ended the game, but has lost
            else:
                # If the user clicks the x, end the game
                running = False
                # for event in pygame.event.get():
                #     if event.type == pygame.QUIT:
                #         running = False
                # # Display the background image with the user score in the middle of the screen
                # screen.blit(backgroundImg, (0, 0))
                # font = pygame.font.SysFont(None, 100)
                # img = font.render("Score: " + str(Score.curScore), True, pygame.Color(255, 255, 255))
                # screen.blit(img, (500, 250))

            # Flip and show the changes to the display
            pygame.display.flip()
            pygame.display.update()
        scores.append(curRewards)
    print(scores)
    first = scores.index(max(scores))
    scores.pop(first)
    second = scores.index(max(scores))
    Models.save_models(models[first], models[second])
# Quit the game
pygame.quit()
