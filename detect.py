#!/usr/bin/python

import os, sys
import math
import datetime
from opencv import cvPoint, cvScalar, cvRectangle, cvSize
from opencv import cvMatchTemplate, cvMinMaxLoc, cvCreateImage, cvCopy, cvNormalize, cvConvertScale
from opencv import IPL_DEPTH_32F, CV_TM_SQDIFF, CV_TM_SQDIFF_NORMED, CV_FILLED, CV_MINMAX, IPL_DEPTH_8U
from opencv.highgui import cvLoadImage, cvSaveImage, CV_LOAD_IMAGE_COLOR
from opencv import adaptors
import pyffmpeg
import pygame
import pygame.image
import pygame.draw
import sqlite3

def here(*args):
    """Get an absolute path by supplying a relative one (relative to this file)."""
    return os.path.join(os.path.dirname(__file__), *args)

def total_seconds(td):
    """timedelta.total_seconds for python < 2.7"""
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6

#
# Constants
#

DEBUG = True

# When to start analyzing the video in seconds
START = 7*60 + 44 #21 #2*60 + 8
# When to end
END = 1*60*60 + 10*60 + 28 #1*60*60 + 54 #55*60 + 22
# How many seconds to skip in each iteration
INTERVAL = 0.25
# Holds the current frame number at all times
CURRENT_FRAME = 0

# Cueballs that have traveled less than this distance are considered not moving (px)
CUEBALL_CONFIRM_DISTANCE = 1.0
# If the cueball hasn't travelled at least this much, it is considered not to have moved (px)
CUEBALL_NOMOVEMENT_DISTANCE = 3.0
# Maximum SQDIFF value that is considered a pocket
POCKET_TRESHOLD = 2000000

# How many possible cueballs to find
NUM_BALLS = 5

# Input video file
VIDEO_FILE = here("assets/session3.mp4")

# Template image paths
BALL_TMPL = here("assets/cueball_smaller.png")
POCKET_TL_TMPL = here("assets/pocket_tl.png")
POCKET_TR_TMPL = here("assets/pocket_tr.png")
POCKET_BL_TMPL = here("assets/pocket_bl.png")
POCKET_BR_TMPL = here("assets/pocket_br.png")

#
# Initialize some variables and database
#

tmpl_ball = None # Ball template image
pocket_templates = {} # dict of pocket template images (keys: tl, tr, bl, br)
pocket_markers = {} # dict of pocket marker positions found in the templates (keys: tl, tr, bl, br)

# We store coordinates here
out_file = None

#
# Classes
#

class Cueball(object):
    """A class representing a cueball. Stores its position, the match value (lower is better) and
    if calculated, the confidence level that this really is the cueball (higher is better) [0..1]."""
    def __init__(self, x=0, y=0, match_value=0):
        self.x = x
        self.y = y
        self.match_value = match_value
        self.confidence = 0
        self.confirmed = False
        self.flags = {} # Custom flags for debugging purposes

    def __str__(self):
        return "[ Cueball %d,%d confidence=%.2f, match_value=%d %s]" % \
               (self.x, self.y, self.confidence, self.match_value, self.confirmed and "CONFIRMED " or "")


# We use this to confirm a cueball standing still
last_best_cueball = None
# We use this not to register the same standing cueball twice
# Initialize it to an impossible cueball to save an if() in the loop
last_confirmed_cueball = Cueball(0, 0, 0)


#
# Utility Functions
#

def normalize(img):
    """Utility function to normalize the image and return the new normalized image using opencv."""
    norm = cvCreateImage(cvSize(img.width, img.height), IPL_DEPTH_32F, 1)
    cvCopy(img, norm)
    cvNormalize(norm, norm, 1, 0, CV_MINMAX)
    norm_u = cvCreateImage(cvSize(img.width, img.height), IPL_DEPTH_8U, 1)
    cvConvertScale(norm, norm_u, 255)
    return norm_u

def findpixel(img, col):
    """Utility function to encapsulate nested for loops so we can use 'return' as a 'break 2' statement,
    because according to Guido this is a very rare and complicated situation (>_<).
    (see http://mail.python.org/pipermail/python-3000/2007-July/008663.html)"""
    for x in range(img.width):
        for y in range(img.height):
            if img[y][x][0] == col[0] and img[y][x][1] == col[1] and img[y][x][2] == col[2]:
                return cvPoint(x, y)

def matchTemplate(image, template):
    """Utility function to match the template against the image and return a heightmap.
    The match is made using V_TM_SQDIFF_NORMED, so smaller values in the heightmap indicate
    closer matches (look for local minima)."""
    match_hmap = cvCreateImage(
        cvSize(image.width-template.width+1, image.height-template.height+1),
        IPL_DEPTH_32F,
        1
    )
    cvMatchTemplate(image, template, match_hmap, CV_TM_SQDIFF_NORMED)
    return match_hmap

def removeMinimum(img, x, y, w, h):
    """Utility function for removing the global minimum from a cvArray.
    This finds the global maximum of the same array and then blits a rectangle of the same color
    over the area designated by x,y and w,h.
    This way to find the next _local_ minimum, we can just remove the current global
    minimum and search for the global one again using cvMinMaxLoc - should be faster than looking
    for all local minima with python."""
    minmax = cvMinMaxLoc(img)
    cvRectangle(img, cvPoint(x-int(w/2), y-int(h/2)), cvPoint(x+int(w/2), y+int(h/2)), cvScalar(minmax[1], 0, 0, 0), CV_FILLED)

#
# Algorithm functions
#

def get_table(image):
    """Find the table in the image by matching each pocket separately.
    Uses POCKET_TRESHOLD and some sanity checks to filter matches.

    Returns a dictionary with keys 'tl', 'tr', 'bl', 'br', with a cvPoint instance
    as the value for each one.
    """
    bounds = {}
    for pocket in ['tl', 'tr', 'bl', 'br']:

        matchmap = matchTemplate(image, pocket_templates[pocket])
        minval, maxval, minloc, maxloc = cvMinMaxLoc(matchmap)

        #print("Pocket %s value = %d" % (pocket, minval))

        if minval < POCKET_TRESHOLD:
            # Found a possible match
            bounds[pocket] = cvPoint(minloc.x + pocket_markers[pocket].x, minloc.y + pocket_markers[pocket].y)
        else:
            # If the best match for a pocket is above the treshold, then we can discard
            # the whole thing
            return None

    if DEBUG: print("Possible table bounds %s" % bounds)

    # Sanity check

    if bounds['tl'].x >= bounds['tr'].x or bounds['bl'].x >= bounds['br'].x:
        return None
    if bounds['tl'].x <= bounds['bl'].x or bounds['tr'].x >= bounds['br'].x:
        return None
    if bounds['bl'].y <= bounds['tl'].y:
        return None

    # Make sure top and bottom edges are straight
    if abs( bounds['tl'].y - bounds['tr'].y ) > image.height / 80:
        return None
    if abs( bounds['bl'].y - bounds['br'].y ) > image.height / 80:
        return None

    # Make sure table is more or less symmetrical
    if math.fabs((bounds['tl'].x-bounds['bl'].x)-(bounds['br'].x-bounds['tr'].x)) > image.width / 100:
        return None

    return bounds

def evaluate_cueball(ball, image, table):
    """Return a confidence level that the ball found is indeed the cueball.
    0 means that the ball is definitely not a cueball.
    A poor algorithm but works well enough."""

    # evaluate ball position
    # this doesn't produce a confidence, but return 0 straight away
    # if the ball is not on the table

    ball.flags['NOT_ON_TABLE'] = True # Remove this afterwards
    # Check that the ball is on the table vertically
    # at the bottom edge, account for the fact that ball.y is the top edge of the ball
    if not table['tl'].y < ball.y < table['bl'].y-tmpl_ball.height-2:
        return 0

    # Check that the ball is on the table horizontally (approximate)
    # Check bottom corners of the table since their x2-x1 is larger
    if not table['bl'].x < ball.x < table['br'].x:
        return 0

    # Check that the ball is on the table horizontally (exact)
    # Calculate y = kx+b lines for the table left and right edges
    left_k = (table['tl'].y-table['bl'].y) / (table['tl'].x-table['bl'].x)
    left_b = table['bl'].y - left_k * table['bl'].x
    right_k = (table['br'].y-table['tr'].y) / (table['br'].x-table['tr'].x)
    right_b = table['br'].y - right_k * table['br'].x

    # Compare using derived x = (y-b)/k
    if not ball.x > (ball.y-left_b)/left_k:
        return 0
    if not ball.x < (ball.y-right_b)/right_k:
        return 0

    ball.flags['NOT_ON_TABLE'] = False

    # evaluate template difference from image

    match_confidence = 1.0 - ball.match_value
    #print(" - match confidence %.2f" % match_confidence)

    # evaluate ball color, take a pixel from strategically chosen points on the ball
    # These points should be white for the white ball and as much different from white as possible
    # for other balls. Somewhere around the center and a bit to the sides will suffice

    col = image[int(ball.y+tmpl_ball.height-1)][ball.x+3]
    col = (col[0], col[1], col[2]) # Convert to tuple for min(), max() use

    # Find the difference from a perfect white
    color_confidence_1 = 1 - (float(0xff-col[0])/255.0 + float(0xff-col[1])/255.0 + float(0xff-col[2])/255.0) / 3.0
    # Any color bias lessens confidence
    color_confidence_1 = ( color_confidence_1 + (1 - min(1, float(max(col)-min(col))/50.0)) ) / 2.0

    col = image[int(ball.y+tmpl_ball.height-1)][ball.x-3]
    col = (col[0], col[1], col[2]) # Convert to tuple for min(), max() use

    color_confidence_2 = 1 - (float(0xff-col[0])/255.0 + float(0xff-col[1])/255.0 + float(0xff-col[2])/255.0) / 3.0
    color_confidence_2 = ( color_confidence_2 + (1 - min(1, float(max(col)-min(col))/50.0)) ) / 2.0

    color_confidence = (color_confidence_1 + color_confidence_2) / 2.0

    #print(" - color confidence %.2f" % color_confidence)

    return (color_confidence+match_confidence) / 2.0

def find_cueballs(image, table):
    """Finds N best cueballs, evaluates them an returns a list of the
    cueballs sorted by best to worst."""
    global CURRENT_FRAME

    matchmap = matchTemplate(image, tmpl_ball)

    cueballs = []
    for i in range(NUM_BALLS):

        minval, maxval, minloc, maxloc = cvMinMaxLoc(matchmap)

        if DEBUG: print("Found %dth cueball at min %d:%d = %.3f, max %d:%d = %.3f" % 
                        (i, minloc.x + int(tmpl_ball.width/2), minloc.y, minval, maxloc.x, maxloc.y, maxval))

        # Ball position is set as the top center of the ball template
        ball = Cueball(minloc.x + int(tmpl_ball.width/2), minloc.y, minval)
        ball.flags['nth_match'] = i
        ball.confidence = evaluate_cueball(ball, image, table)

        cueballs.append(ball)

        # Remove the local minimum so we can use cvMinMaxLoc again
        removeMinimum(matchmap, minloc.x, minloc.y, tmpl_ball.width, tmpl_ball.height)

    # We are able to save a pretty interesting normalized heightmap of the
    # match here
    #cvSaveImage(here("results/%d_hmap_ball.jpg" % CURRENT_FRAME), normalize(matchmap))

    cueballs = sorted(cueballs, key=lambda x: x.confidence)
    cueballs.reverse()

    return cueballs

def draw_interesting_stuff(image, table, cueballs):
    """Draw table bounds, found cueballs and other interesting
    data to image."""

    img_pygame = pygame.image.frombuffer(image.tostring(), image.size, image.mode)
    font = pygame.font.Font(pygame.font.get_default_font(), 12)

    if table:
        for pocket in ['tl', 'tr', 'bl', 'br']:
            # Draw pockets
            pygame.draw.rect(
                img_pygame,
                 (0xff, 0xbb, 0x44),
                 pygame.Rect(
                     table[pocket].x-pocket_markers[pocket].x,
                     table[pocket].y-pocket_markers[pocket].y,
                     pocket_templates[pocket].width,
                     pocket_templates[pocket].height
                ),
                1
            )
            # Draw table bounds
            pygame.draw.lines(
                img_pygame,
                (0xdd, 0xaa, 0x00),
                True,
                [
                    (table['tl'].x, table['tl'].y),
                    (table['tr'].x, table['tr'].y),
                    (table['br'].x, table['br'].y),
                    (table['bl'].x, table['bl'].y),
                ],
                1
            )

    if cueballs:
        for cueball in cueballs:
            pygame.draw.rect(
                img_pygame,
                cueball is cueballs[0] and (0xcc, 0x22, 0x00) or (0x88, 0x88, 0x88),
                pygame.Rect(
                    cueball.x-int(tmpl_ball.width/2)-7,
                    cueball.y-7,
                    tmpl_ball.width+14,
                    tmpl_ball.height * 2 + 16
                ),
                cueball.confirmed and 3 or 1
            )
            # Display confidence
            img_pygame.blit(
                font.render("%.2f" % cueball.confidence, True, (0xff, 0xff, 0xff)),
                (cueball.x-int(tmpl_ball.width/2)-5, cueball.y-7-font.get_height())
            )
            # Display if ball is not on table
            if cueball.flags.get('NOT_ON_TABLE'):
                img_pygame.blit(
                    font.render("NOT", True, (0xcc, 0x00, 0x00)),
                    (cueball.x-int(tmpl_ball.width/2)-5, cueball.y+tmpl_ball.height*2+10)
                )
            # Display the match number. If the ball is incorrectly identified as not-a-cueball but this
            # number is 0, then it is the evaluator's fault for dismissing it on other bases.
            img_pygame.blit(
                font.render(str(cueball.flags['nth_match']), True, (0xcc, 0xcc, 0xcc)),
                (cueball.x-int(tmpl_ball.width/2)-18, cueball.y-7)
            )



    return img_pygame

def store_results(image, table, cueballs):
    """Save any images and other data."""
    global CURRENT_FRAME, out_file
    #image.save('results/%d.jpg' % frame_no)
    if cueballs:
        best_ball = cueballs[0]
        if best_ball.confirmed:
            pygame.image.save(image, here('results/%d.jpg' % CURRENT_FRAME))
            out_file.write("%d %d %d\n" % (CURRENT_FRAME, best_ball.x, best_ball.y))

def main():
    """Run the whole shiz."""
    global tmpl_ball, pocket_templates, pocket_markers, last_best_cueball, last_confirmed_cueball, CURRENT_FRAME, out_file

    start_time = datetime.datetime.now()

    # Open output files

    out_file = open(here("confirmed_cueballs.txt"), 'w')

    # Initialize pygame

    pygame.init()
    pygame.display.set_caption("Snooker - main info window")
    window = pygame.display.set_mode((704, 400))

    # Load and initialize template images

    tmpl_ball = cvLoadImage(BALL_TMPL, CV_LOAD_IMAGE_COLOR)
    if not tmpl_ball:
        raise Exception("Failed to load ball template image")

    # Load pocket templates, find marker pixel in template and remember it

    for pocket in ['tl', 'tr', 'bl', 'br']: 
        pocket_tmpl = cvLoadImage(globals()['POCKET_%s_TMPL' % pocket.upper()], CV_LOAD_IMAGE_COLOR)
        if not pocket_tmpl:
            raise Exception("Failed to load pocket '%s' template" % pocket)
        pocket_templates[pocket] = pocket_tmpl
        pocket_markers[pocket] = findpixel(pocket_tmpl, cvScalar(255, 0, 255))
        if pocket_markers[pocket]:
            if DEBUG: print("Found marker for pocket %s at %d:%d" % 
                            (pocket.upper(), pocket_markers[pocket].x, pocket_markers[pocket].y))

    stream = pyffmpeg.VideoStream()

    stream.open(VIDEO_FILE)

    fps = stream.tv.get_fps()
    duration_f = stream.tv.duration()

    start_frame = int(START*fps)
    end_frame = int(END*fps)
    interval = int(INTERVAL*fps)

    length = float(end_frame-start_frame)/float(fps) # Seconds

    i = start_frame

    while i < end_frame:
        CURRENT_FRAME = i

        # Calculate info about the total progress and display it

        percentage = float(i-start_frame)/float(end_frame-start_frame)*100.0
        estimated_length = 0
        if i > start_frame:
            estimated_length = float(total_seconds((datetime.datetime.now()-start_time))) / percentage * 100.0

        print("Frame #%d (%d-%d) %.1f%%" % (i, start_frame, end_frame, percentage))
        pygame.display.set_caption("Snooker - main info window - Frame %d - %.1f%% (est. total %.1fmin)" % (i, percentage, estimated_length/60.0))

        image = stream.GetFrameNo(i)
        image_ipl = adaptors.PIL2Ipl(image)

        table = get_table(image_ipl)
        cueballs = None
        best_ball = None

        if table:
            cueballs = find_cueballs(image_ipl, table)

        if cueballs:

            # See if the best cueball we found now, is the best cueball found in the previous run,
            # which would mean that it's standing still. Only consider the best matches.

            if not last_best_cueball:
                last_best_cueball = cueballs[0]
            else:

                distance_to_last = math.sqrt( (last_best_cueball.x-cueballs[0].x)**2 + (last_best_cueball.y-cueballs[0].y)**2 )
                distance_to_last_confirmed = math.sqrt( (last_confirmed_cueball.x-cueballs[0].x)**2 + 
                                                        (last_confirmed_cueball.y-cueballs[0].y)**2 )

                if distance_to_last < CUEBALL_CONFIRM_DISTANCE and distance_to_last_confirmed > CUEBALL_NOMOVEMENT_DISTANCE:
                    if cueballs[0].confidence >= 0.72: # Extra check to remove most of the false positives
                        # CONFIRMED
                        last_confirmed_cueball = cueballs[0]
                        cueballs[0].confirmed = True
                        if DEBUG: print("CONFIRMED cueball %s" % cueballs[0])
                else:
                    if DEBUG: print("Dropped the ball because distance condition failed: %.2f<%d and %.2f>%d" % 
                                    (distance_to_last, CUEBALL_CONFIRM_DISTANCE, 
                                     distance_to_last_confirmed, CUEBALL_NOMOVEMENT_DISTANCE))

                # If confirmation or sequence fail, reset and try the whole thing again
                last_best_cueball = False

        image_interesting = draw_interesting_stuff(image, table, cueballs)

        window.blit(image_interesting, (0,0))
        pygame.display.flip()

        store_results(image_interesting, table, cueballs)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                out_file.close()
                sys.exit(0)
            else:
                pass
                #print event

        # Skip <interval> frames
        i += interval

    out_file.close()

if __name__ == "__main__":
    main()
