import sys, pygame

from pygame.locals import *


def draw():
    pygame.init()
    brush_size = 50
    screen = pygame.display.set_mode((560,560))
    screen.fill((0,0,0))
    brush = pygame.image.load("brush.png")
    brush = pygame.transform.scale(brush, (brush_size, brush_size))

    pygame.display.update()
    clock = pygame.time.Clock()
    mouse_down = False
    while True:
        clock.tick(200)
        x, y = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    pygame.image.save(screen, "screenshot.png")
                    pygame.quit()
                    return
            elif event.type == MOUSEBUTTONDOWN:
                mouse_down = True
            elif event.type == MOUSEBUTTONUP:
                mouse_down = False


            if(mouse_down):
                screen.blit(brush, (x-brush_size/2, y-brush_size/2))
                pygame.display.update()
