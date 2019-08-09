import pygame
import neuralnetwork
import my_matrix_lib
pygame.init()

nn = neuralnetwork.load_json("mnist_nn.json")
white = (255, 255, 255)
black = (0, 0, 0)
size = 280, 280
stroke = 10
input_list = []

screen = pygame.display.set_mode(size)
pygame.display.set_caption("Mnist Neural Network")
screen.fill(black)


while True:
    # function for Pygame updating the screen
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            array = pygame.PixelArray(screen)
            for i in range(0, len(array), 10):
                for j in range(0, len(array), 10):
                    if array[j][i] == 0:
                        input_list.append(0)
                    else:
                        input_list.append(1)
            output = nn.feed_forward(input_list)
            print(my_matrix_lib.Matrix.arg_max(output))
            screen.fill(black)
            input_list = []
    if pygame.mouse.get_pressed()[0] == 1:
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(screen, white, mouse_pos, stroke)
    pygame.display.update()
