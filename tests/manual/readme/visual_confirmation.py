from investigators.visuals import cv2  # package includes a fallback cv2
from investigators.visuals import TemplateFinder, ProportionalRegion

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

# clean up
cv2.waitKey()
cv2.destroyAllWindows()
