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


Command Line Example for currently working investigators
================

This can be found in tests/manual/.

    from investigators.visuals import cv2  # package includes a fallback cv2
    from investigators.visuals import TemplateFinder, ProportionalRegion, Grid
    from investigators.visuals import ImageIdentifier

    # Show the starting screenshot
    screenshot = cv2.imread('screenshot 1280x800, game 800x600.png')
    cv2.imshow('screenshot', screenshot)

    # Use TemplateFinder to locate and show an image of the game on the screen
    game_template = cv2.imread('game 640x480.png')
    game_mask = cv2.imread('game mask 640x480.png')
    game_possible_sizes = (600, 800), (240, 320)
    imgf = TemplateFinder(game_template, sizes=game_possible_sizes, mask=game_mask,
                          acceptable_threshold=0.5,
                          immediate_threshold=0.1)
    game_top, game_left, game_bottom, game_right = imgf.locate_in(screenshot)
    game = screenshot[game_top:game_bottom, game_left:game_right]
    cv2.waitKey()
    cv2.imshow('extracted game', game)

    # Use ProportionalRegion to isolate the board within the game
    board_top_proportion = float(80) / 480  # these are just from inspection
    board_left_proportion = float(135) / 640
    board_bottom_proportion = float(450) / 480
    board_right_proportion = float(504) / 640
    pr = ProportionalRegion((board_top_proportion, board_left_proportion,
                            board_bottom_proportion, board_right_proportion))
    board_top, board_left, board_bottom, board_right = pr.region_in(game)
    board = game[board_top:board_bottom, board_left:board_right]
    cv2.waitKey()
    cv2.imshow('extracted board', board)

    # Use Grid to split the grid into image cells. record only the first four
    board_dimensions = 8, 8
    tile_padding = (0.1, 0.1, 0.1, 0.1)
    grid = Grid(board_dimensions, tile_padding)
    first_four_tiles = list()
    for grid_p, cell_borders in grid.borders_by_grid_position(board):
        if grid_p in ((0, 0), (0, 1), (0, 2), (0, 3)):
            top, left, bottom, right = (cell_borders.top, cell_borders.left,
                                        cell_borders.bottom, cell_borders.right)
            first_four_tiles.append(board[top:bottom, left:right].copy())

    # Use ImageIdentifier to identify each of the first four tiles
    ii = ImageIdentifier({'red': cv2.imread('r.png'),
                          'blue': cv2.imread('b.png'),
                          'coins': cv2.imread('m.png'),
                          'skull': cv2.imread('s.png')})
    for tile in first_four_tiles:
        name = ii.identify(tile)
        cv2.putText(tile, name, (2, 20), 0, 0.6, (255, 255, 255), 2)
        cv2.waitKey()
        cv2.imshow('identified original tile', tile)

    # clean up
    cv2.waitKey()
    cv2.destroyAllWindows()


License
=======
MIT. See LICENSE
