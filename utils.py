import argparse
import os
import torch
import torch.nn.functional as F


import numpy as np




def cosine_similarity(x,y):
    assert x.shape[-1] == y.shape[-1]
    # x= x/x.norm()
    # y = y/y.norm()
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    return x@y.transpose(-2,-1)

def euclidean_dist( x, y, normalize=False):
    # x: N x D
    # y: M x D
    if normalize:
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    # return x@y.T

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    x = x.to(y.dtype)
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)#correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--dataset'     , default='ucf101')
    parser.add_argument('--method',       default='taskadapter')
    parser.add_argument('--train_n_way' , default=5, type=int)
    parser.add_argument('--test_n_way'  , default=5, type=int) 
    parser.add_argument('--n_shot'      , default=1, type=int) 
    parser.add_argument('--num_gpus'      , default=2, type=int,help=" number of GPUs to distribute model") 
    parser.add_argument('--train_aug'   , default=True, type=bool)
    parser.add_argument('--work_dir'     , default= os.path.join(os.path.abspath('.'),'workspace'))
    parser.add_argument('--num_segments', default=8, type=int)
    parser.add_argument('--n_query', default=1, type=int)
    parser.add_argument('--eval_freq', default=30, type=int)
    parser.add_argument('--train_episode', default=1000, type=int)# if SSv2 
    parser.add_argument('--eval_episode', default=1000, type=int)
    parser.add_argument('--test_episode', default=10000, type=int)
    parser.add_argument('--test_model', default=False, type=bool)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--save_freq'   , default=100, type=int)
    parser.add_argument('--start_epoch' , default=0, type=int)
    parser.add_argument('--stop_epoch'  , default=10, type=int)
    parser.add_argument('--lr', default = 1e-3, type=float)#1e-5
    parser.add_argument('--message', '-m', default = "nothing to say", type=str, help="same as git's '-m'")
    
    return parser.parse_args()



hmdb_cls = ['brush_hair', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword', 'dribble', 'drink', 'fall_floor', 'flic_flac', 'handstand', 'hug', 'jump', 'kiss', 'pullup', 'punch', 'push', 'ride_bike', 'ride_horse', 'shake_hands', 'shoot_bow', 'situp', 'stand', 'sword', 'sword_exercise', 'throw', 'turn', 'walk', 'wave', 'cartwheel', 'eat', 'golf', 'hit', 'laugh', 'shoot_ball', 'shoot_gun', 'smile', 'somersault', 'swing_baseball', 'fencing', 'kick', 'kick_ball', 'pick', 'pour', 'pushup', 'run', 'sit', 'smoke', 'talk']
hmdb_cls = ["a photo of action about "+i for i in hmdb_cls]

ucf_cls = ['ApplyEyeMakeup', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CricketBowling', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FrisbeeCatch', 'FrontCrawl', 'Haircut', 'Hammering', 'HeadMassage', 'HulaHoop', 'JavelinThrow', 'JugglingBalls', 'JumpingJack', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDhol', 'PlayingFlute', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PullUps', 'PushUps', 'Rafting', 'RopeClimbing', 'Rowing', 'ShavingBeard', 'Skijet', 'SoccerJuggling', 'SoccerPenalty', 'SumoWrestling', 'Swing', 'TableTennisShot', 'TaiChi', 'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo', 'ApplyLipstick', 'CricketShot', 'HammerThrow', 'HandstandPushups', 'HighJump', 'HorseRiding', 'PlayingDaf', 'PlayingGuitar', 'Shotput', 'SkateBoarding', 'BlowingCandles', 'CleanAndJerk', 'CliffDiving', 'CuttingInKitchen', 'Diving', 'FloorGymnastics', 'GolfSwing', 'HandstandWalking', 'HorseRace', 'IceDancing', 'JumpRope', 'PommelHorse', 'Punch', 'RockClimbingIndoor', 'SalsaSpin', 'Skiing', 'SkyDiving', 'StillRings', 'Surfing', 'TennisSwing', 'VolleyballSpiking']
ucf_cls = ["a photo of action about "+i for i in ucf_cls]

smsm_cls = ['Pouring something into something', 'Poking a stack of something without the stack collapsing', 'Pretending to poke something', 'Lifting up one end of something without letting it drop down', 'Moving part of something', 'Moving something and something away from each other', 'Removing something, revealing something behind', 'Plugging something into something', 'Tipping something with something in it over, so something in it falls out', 'Stacking number of something', "Putting something onto a slanted surface but it doesn't glide down", 'Moving something across a surface until it falls down', 'Throwing something in the air and catching it', 'Putting something that cannot actually stand upright upright on the table, so it falls on its side', 'Holding something next to something', 'Pretending to put something underneath something', "Poking something so lightly that it doesn't or almost doesn't move", 'Approaching something with your camera', 'Poking something so that it spins around', 'Pushing something so that it falls off the table', 'Spilling something next to something', 'Pretending or trying and failing to twist something', 'Pulling two ends of something so that it separates into two pieces', 'Lifting up one end of something, then letting it drop down', "Tilting something with something on it slightly so it doesn't fall down", 'Spreading something onto something', 'Touching (without moving) part of something', 'Turning the camera left while filming something', 'Pushing something so that it slightly moves', 'Uncovering something', 'Moving something across a surface without it falling down', 'Putting something behind something', 'Attaching something to something', 'Pulling something onto something', 'Burying something in something', 'Putting number of something onto something', 'Letting something roll along a flat surface', 'Bending something until it breaks', 'Showing something behind something', 'Pretending to open something without actually opening it', 'Pretending to put something onto something', 'Moving away from something with your camera', 'Wiping something off of something', 'Pretending to spread air onto something', 'Holding something over something', 'Pretending or failing to wipe something off of something', 'Pretending to put something on a surface', 'Moving something and something so they collide with each other', 'Pretending to turn something upside down', 'Showing something to the camera', 'Dropping something onto something', "Pushing something so that it almost falls off but doesn't", 'Piling something up', 'Taking one of many similar things on the table', 'Putting something in front of something', 'Laying something on the table on its side, not upright', 'Lifting a surface with something on it until it starts sliding down', 'Poking something so it slightly moves', 'Putting something into something', 'Pulling something from right to left', 'Showing that something is empty', 'Spilling something behind something', 'Letting something roll down a slanted surface', 'Holding something behind something', 'Lifting something up completely without letting it drop down', 'Pouring something into something until it overflows', 'Putting something, something and something on the table', 'Trying to bend something unbendable so nothing happens', 'Pouring something out of something', 'Throwing something onto a surface', 'Putting something onto something else that cannot support it so it falls down', 'Pretending to pour something out of something, but something is empty', 'Pulling something out of something', 'Holding something in front of something', 'Tilting something with something on it until it falls off', 'Moving something away from the camera', 'Twisting (wringing) something wet until water comes out', 'Poking a hole into something soft', 'Pretending to take something from somewhere', 'Putting something upright on the table', 'Poking a hole into some substance', 'Rolling something on a flat surface', 'Poking a stack of something so the stack collapses', 'Twisting something', 'Something falling like a feather or paper', 'Putting something on the edge of something so it is not supported and falls down', 'Pushing something off of something', 'Dropping something into something', 'Letting something roll up a slanted surface, so it rolls back down', 'Pushing something with something', 'Opening something', 'Putting something on a surface', 'Taking something out of something', 'Spinning something that quickly stops spinning', 'Unfolding something', 'Moving something towards the camera', 'Putting something next to something', 'Scooping something up with something', 'Squeezing something', 'Failing to put something into something because something does not fit']
smsm_cls = ["a photo of action about "+i for i in smsm_cls]

kinetics_cls = ['air drumming', 'arm wrestling', 'beatboxing', 'biking through snow', 'blowing glass', 'blowing out candles', 'bowling', 'breakdancing', 'bungee jumping', 'catching or throwing baseball', 'cheerleading', 'cleaning floor', 'contact juggling', 'cooking chicken', 'country line dancing', 'curling hair', 'deadlifting', 'doing nails', 'dribbling basketball', 'driving tractor', 'drop kicking', 'dying hair', 'eating burger', 'feeding birds', 'giving or receiving award', 'hopscotch', 'jetskiing', 'jumping into pool', 'laughing', 'making snowman', 'massaging back', 'mowing lawn', 'opening bottle', 'playing accordion', 'playing badminton', 'playing basketball', 'playing didgeridoo', 'playing ice hockey', 'playing keyboard', 'playing ukulele', 'playing xylophone', 'presenting weather forecast', 'punching bag', 'pushing cart', 'reading book', 'riding unicycle', 'shaking head', 'sharpening pencil', 'shaving head', 'shot put', 'shuffling cards', 'slacklining', 'sled dog racing', 'snowboarding', 'somersaulting', 'squat', 'surfing crowd', 'trapezing', 'using computer', 'washing dishes', 'washing hands', 'water skiing', 'waxing legs', 'weaving basket', 'baking cookies', 'crossing river', 'dunking basketball', 'feeding fish', 'flying kite', 'high kick', 'javelin throw', 'playing trombone', 'scuba diving', 'skateboarding', 'ski jumping', 'trimming or shaving beard', 'blasting sand', 'busking', 'cutting watermelon', 'dancing ballet', 'dancing charleston', 'dancing macarena', 'diving cliff', 'filling eyebrows', 'folding paper', 'hula hooping', 'hurling (sport)', 'ice skating', 'paragliding', 'playing drums', 'playing monopoly', 'playing trumpet', 'pushing car', 'riding elephant', 'shearing sheep', 'side kick', 'stretching arm', 'tap dancing', 'throwing axe', 'unboxing']
kinetics_cls = ["a photo of action about "+i for i in kinetics_cls]