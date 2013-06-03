investigators
========

Achieve various investigations of data by combining easy-to-use
investigator classes.

For example:

- TemplateFinder (find a template within an image)
- ProportionalRegion (use proportions rather than pixels to get a view)
- Grid (split up an image into cells)
- ImageIdentifier (identify a given image from a library of examples)


Status
======

TemplateFinder: working
ProportionalRegion: working


Command Line Example
================

This is long, but shows all of the currently working investigators.

It can be found along with the test images in tests/manual/.

    import numpy
    from investigators.visuals import cv2  # package includes a fallback cv2
    from investigators.visuals import TemplateFinder, ProportionalRegion, Grid
    from investigators.visuals import ImageIdentifier, TankLevel

    # Show the starting screenshot
    screenshot = cv2.imread('screenshot 1280x800, game 800x600.png')
    cv2.imshow('screenshot', screenshot)
    cv2.waitKey()

    # Use TemplateFinder to locate and show an image of the game on the screen
    template = cv2.imread('game 640x480.png')
    mask = cv2.imread('game mask 640x480.png')
    possible_sizes = (600, 800), (240, 320)
    finder = TemplateFinder(template, sizes=possible_sizes, mask=mask,
                            scale_for_speed=0.25)
    t, l, b, r = finder.locate_in(screenshot)
    game = screenshot[t:b, l:r]
    cv2.destroyWindow('screenshot')
    cv2.imshow('extracted game', game)
    cv2.waitKey()

    # Use ProportionalRegion to isolate the board within the game
    tlbr_proportions = (80.0 / 480, 135.0 / 640, 450.0 / 480, 504.0 / 640)
    proportional = ProportionalRegion(tlbr_proportions)
    t, l, b, r = proportional.region_in(game)
    board = game[t:b, l:r]
    cv2.imshow('extracted board', board)
    cv2.waitKey()

    # Use Grid to split the grid into image cells. record only the first four
    board_dimensions = 8, 8
    tile_padding = (0.1, 0.1, 0.1, 0.1)  # just for demonstration
    grid = Grid(board_dimensions, tile_padding)
    first_four_tiles = list()
    for grid_p, cell_borders in grid.borders_by_grid_position(board):
        if grid_p in ((0, 0), (0, 1), (0, 2), (0, 3)):
            t, l, b, r = cell_borders
            first_four_tiles.append(board[t:b, l:r])

    # Use ImageIdentifier to identify each of the first four tiles
    identifier = ImageIdentifier({'red': cv2.imread('r.png'),
                                  'blue': cv2.imread('b.png'),
                                  'coins': cv2.imread('m.png'),
                                  'skull': cv2.imread('s.png')})
    for tile in first_four_tiles:
        name = identifier.identify(tile)
        cv2.putText(tile, name, (2, 20), 0, 0.6, (255, 255, 255), 2)
        cv2.imshow('identified original tile', tile)
        cv2.waitKey()
    cv2.destroyWindow('identified original tile')
    cv2.destroyWindow('extracted board')

    # Use ProportionalRegion to isolate the player's health within the game
    tlbr_proportions = (76.0 / 480, 8.0 / 640, 87.0 / 480, 116.0 / 640)
    proportional = ProportionalRegion(tlbr_proportions)
    t, l, b, r = proportional.region_in(game)
    health = game[t:b, l:r]

    # Use TankLevel to measure how full the health bar is
    health_color = (4, 4, 110)
    empty_health_color = (15, 15, 50)
    ignore_color = (0, 0, 0)
    tank_level = TankLevel(health_color, empty_health_color, ignore_color)
    health = numpy.rot90(health)  # make it upright for the TankLevel measurement
    health_pcnt = 100.0 * tank_level.how_full(health)
    health_string = '~{:2.0f}%'.format(health_pcnt)
    health = numpy.rot90(health, -1)  # rotate and enlarge for display
    h, w = health.shape[0:2]
    health = cv2.resize(health, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    cv2.putText(health, health_string, (25, 25), 0, 0.8, (0, 255, 0), 2)
    cv2.imshow('player health bar', health)
    cv2.waitKey()

    # Use ProportionalRegion to isolate the extra actions area within the game
    tlbr_proportions = (120.0 / 960, 1100.0 / 1280, 160.0 / 960, 1220.0 / 1280)
    proportional = ProportionalRegion(tlbr_proportions)
    t, l, b, r = proportional.region_in(game)
    token_region = game[t:b, l:r]

    # Use TemplateFinder (multiple) to check for extra actions
    token_template = cv2.imread('extra_turn_token.png')
    possible_sizes = (17, 14),
    finder = TemplateFinder(token_template, sizes=possible_sizes,
                            acceptable_threshold=0.1, immediate_threshold=0.1)
    found_tokens = finder.locate_multiple_in(token_region)
    for borders in found_tokens:
        t, l, b, r = borders
        cv2.rectangle(token_region, (l, t), (r, b), (255, 255, 255), 2)
    h, w = token_region.shape[0:2]
    token_region = cv2.resize(token_region, (w * 3, h * 3),
                              interpolation=cv2.INTER_CUBIC)
    cv2.imshow('extra actions', token_region)
    cv2.waitKey()

    # clean up
    cv2.destroyAllWindows()


License
=======
MIT. See LICENSE
