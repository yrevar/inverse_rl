import math
import pygame
import random
from pygame.locals import *

white = (255,255,255)

DEBUG=False

_center_loc = lambda obj, loc=(0,0): (loc[0] - obj.get_width()/2., loc[1] - obj.get_height()/2.)
class Point:
    """
    Adapted from: https://codereview.stackexchange.com/questions/70143/drawing-a-dashed-line-with-pygame
    """
    # constructed using a normal tupple
    def __init__(self, point_t = (0,0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])
    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))
    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))
    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))
    def __div__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))
    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))
    def dist(self, other, kind='euclidean'):
        if kind == 'euclidean':
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        else:
            return abs(self.x - other.x) + abs(self.y - other.y)
    # get back values in original tuple format
    def get(self):
        return (self.x, self.y)

def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length

    for index in range(0, length/dash_length, 2):
        start = origin + (slope *    index    * dash_length)
        end   = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)

def bounded_sigmoid(x, y_min, y_max, x_mid, steep):
    """
    Experimented here: https://www.desmos.com/calculator/1yekcguofk 
    """
    return y_min + (y_max-y_min) / (1 + math.exp(steep*(-x + x_mid)))

def is_clicked(key, keys, prev_keys):

    if prev_keys[key] and not keys[key]: # key pressed and released
        return True
    return False

def is_pressed(key, keys, prev_keys):

    if not prev_keys[key] and keys[key]: # key pressed and released
        return True
    return False

class Car(pygame.sprite.Sprite):

    def __init__(self, img_normal, img_accident, position, velocity):

        # Call the parent class (Sprite) constructor
        super(Car, self).__init__()
        self.img_normal = img_normal
        self.img_accident = img_accident
        self.image = pygame.image.load(self.img_normal)
        self.orig_image = self.image
        self.rect = self.image.get_rect()

        self.h = self.image.get_height()
        self.w = self.image.get_width()
        self.rect.x, self.rect.y = _center_loc(self.image, position)
        self.dx = velocity[0]
        self.dy = velocity[1]
        self.new_x = self.rect.x
        self.new_y = self.rect.y
        self.collided = False
        self.x_diff = 0
        self.y_diff = 0

    def get_image(self):
        return self.image

    def get_loc(self):
        return self.rect.x, self.rect.y

    def set_velocity(self, dx=None, dy=None):
        if dx is not None:
            self.dx = dx
        if dy is not None:
            self.dy = dy

    def add_velocity(self, dx=None, dy=None):
        if dx is not None:
            self.dx += dx
        if dy is not None:
            self.dy += dy

    def chase_target(self):

        if self.rect.x != self.new_x:
            self.x_diff = self.new_x - self.rect.x
            sign = 1 if self.x_diff > 0 else -1
            if abs(self.x_diff) <= self.dx:
                self.rect.x = self.new_x
                self.image = pygame.transform.rotate(self.orig_image, 0)
            else:
                self.rect.x += self.dx if self.x_diff > 0 else -self.dx
                self.image = pygame.transform.rotate(self.orig_image,
                                                     -1 * sign * bounded_sigmoid(abs(self.x_diff), 0, 45, 22, 0.25))

        if self.rect.y != self.new_y:
            self.y_diff = self.new_y - self.rect.y
            if abs(self.y_diff) <= self.dy:
                self.rect.y = self.new_y
            else:
                self.rect.y += self.dy if self.y_diff > 0 else -self.dy

    def run(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

    def move(self, dx=None, dy=None):

        if dx is not None:
            self.new_x += dx
        if dy is not None:
            self.new_y += dy

    def _collide(self):
        self.image = pygame.image.load(self.img_accident)
        self.orig_image = self.image
        self.collided = True
        audio_hit.play()

    def _fix(self):
        self.collided = False
        self.image = pygame.image.load(self.img_normal)
        self.orig_image = self.image

    def collision_update(self, othercar):
        if self.rect.colliderect(othercar.rect):
            if not self.collided:
                self._collide()

class RoadCarManager(object):

    def __init__(self, normal_img_file, accident_img_file, lanes_x=[240-80, 240, 240+80]):

        self.cars = pygame.sprite.Group()
        self.speed_max = -(MAX_VEL-SPEED_UNIT)
        self.rand_entry_y = lambda: random.randint(-500,-100)
        self.rand_velocity = lambda: random.randint(self.speed_max,-SPEED_UNIT)
        for x in lanes_x:
            self.cars.add(Car(normal_img_file, accident_img_file,
                              (x, self.rand_entry_y()),
                              (0, self.rand_velocity())))

    def run(self, frame_of_reference_dy):
        for i, car_i in enumerate(self.cars):
            car_i.run()
            if car_i.rect.y > height + 200 or car_i.rect.y < -400:
                car_i.rect.y = self.rand_entry_y()
                car_i.set_velocity(0, self.rand_velocity())
                car_i.add_velocity(dy=frame_of_reference_dy)
                if DEBUG: print "new car: ", car_i.rect.y, car_i.dy,
                car_i._fix()
        print ""

    def add_velocity(self, dx=None, dy=None):
        for car_i in self.cars:
            car_i.add_velocity(dx, dy)

    def check_collision(self, target_car):
        hit_list = pygame.sprite.spritecollide(target_car, self.cars, False)
        for car_i in hit_list:
            car_i.collision_update(car)

# Initialize the game
pygame.init()
width, height = 480, 640
SPEED_UNIT = 6
MAX_VEL = SPEED_UNIT * 3
mid_x, mid_y = width/2., height/2.
screen=pygame.display.set_mode((width, height))

# Load images
road = pygame.image.load("resources/images/road.png")
road_loc = (0,0)
car = Car("resources/images/ego_car.png",
          "resources/images/ego_car.png",
          (width/2, height-100), (0, 0))
dash_start_y= 0
dash_len = 30
car_speed = 0
road_car_mgr = RoadCarManager("resources/images/road_car_normal.png",
               "resources/images/road_car_accident.png")

# Music
pygame.mixer.init()
voice = pygame.mixer.Channel(5)
audio_car= pygame.mixer.Sound("resources/audio/CarDriving.wav")
audio_hit = pygame.mixer.Sound("resources/audio/crash_x.wav")
audio_car.set_volume(0.8)
audio_hit.set_volume(0.5)
pygame.mixer.music.load("resources/audio/street-daniel_simon.wav")
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(0.3)

# Prev keys: maintain to detect click events
prev_keys = pygame.key.get_pressed()

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# GUI loop
while not done:

    # --- Clear the screen before drawing it again
    screen.fill(0)

    # --- Loop through the events
    for event in pygame.event.get():
        # check if the event is the X button 
        if event.type==pygame.QUIT or\
            (event.type == pygame.KEYDOWN and event.key== K_q):
            # if it is quit the game
            done = True
        keys = pygame.key.get_pressed()

    # --- Draw the screen elements
    # Draw background
    screen.blit(road, road_loc)

    # --- Process Input
    # Update divider line position (for illusion of a running car)
    if is_clicked(pygame.K_UP, keys, prev_keys): # up clicked
        car_speed += SPEED_UNIT
        if abs(car_speed) <= MAX_VEL:
            road_car_mgr.add_velocity(dy=SPEED_UNIT)
        car_speed = max(min(car_speed, MAX_VEL), -MAX_VEL)

    if is_clicked(pygame.K_DOWN, keys, prev_keys): # down clicked
        car_speed -= SPEED_UNIT
        if abs(car_speed) <= MAX_VEL:
            road_car_mgr.add_velocity(dy=-SPEED_UNIT)
        car_speed = max(min(car_speed, MAX_VEL), -MAX_VEL)

    if car_speed:
        if not voice.get_busy():
            voice.play(audio_car)
        if is_pressed(pygame.K_LEFT, keys, prev_keys): # left clicked
            car.move(dx=-80)
        elif is_pressed(pygame.K_RIGHT, keys, prev_keys): # left clicked
            car.move(dx=80)
        car.set_velocity(dx=2*car_speed, dy=0)
        car.chase_target()
        dash_start_y = (dash_start_y + car_speed) % (2 * dash_len)
    else:
        audio_car.stop()
    prev_keys = keys
    # ---

    road_car_mgr.run(car_speed)
    if DEBUG:
        for car_i in road_car_mgr.cars:
            print "road cars: ", car_i.rect.y, car_i.dy,
    # Lanes
    draw_dashed_line(screen, white, (mid_x-40, dash_start_y), (mid_x-40, height), 5, dash_len)
    draw_dashed_line(screen, white, (mid_x+40, dash_start_y), (mid_x+40, height), 5, dash_len)
    # Ego car
    screen.blit(car.get_image(), car.get_loc())
    # Road cars
    road_car_mgr.cars.draw(screen)
    road_car_mgr.check_collision(car)
    # Update the screen
    pygame.display.flip()

    # Limit to 60 frames per second
    clock.tick(60)
