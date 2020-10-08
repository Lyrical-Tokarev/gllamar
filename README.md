# gllamar
alpaca motionlessly lies and eats some carrots instead of ryes [yet another hackathon project]

## WTF is this

Here you can find [ODS pet projects](https://ods.ai/projects/pet-projects) community's hackathon project.

The main goal: translate human photos to and from alpacas | llamas and other camelids. Because I love them.

Some or all of the code/interactive parts might be in Russian.

## Our Grand Plan

During the hackathon, which is held at October, 2 - October, 10, 2020 we plan to do:
- collect alpaca | llama | other cutie photos
- annotate them (we'll need bounding boxes with animal faces, since image-to-image translation will work better in this case)
- use magic of CycleGAN (most probably we'll use [this implementation](https://github.com/junyanz/CycleGAN))
- box it: make Telegram bot, mobile application or just demo app with streamlit.

## Project limitations

- we might find not enough alpacas to achieve good results. For now we have only 432 images with camelids faces.
- tonz of other exciting projects with upcoming deadlines (like [this one](https://www.kaggle.com/c/stanford-covid-vaccine)) elsewhere, which are hard to resist, especially when you have 1-week long vacation and admire puzzles
- both of my teammates have no vacations now and have to do their work.

## Tools of trade we use

- pytorch
- detectron2
- CycleGAN

- streamlit for visual debug and screencasts

# Data: collection and preprocessing

We use OpenImages dataset, since it provides annotations (both segmentation and bounding boxes) for the class named 'Alpaca'.
Since annotations describe the whole animal bodies, and we've decided to use only animal faces, we've also used labelme to select bounding boxes for faces by hand.

For now we have 432 images with alpaca, llama or guanaco.

## Next steps and/or ideas:

- [x] add "labelme to detectron2" format converter
- [ ] add script for detectron2 training with this preprocessed dataset and use it to make alpaca-detector
- [ ] ? add more images to our dataset from flickr and to extract animal faces with detectron2 trained with OpenImages-based Alpaca subset.
- [ ] validate collected and automatically annotated data
- [ ] add script to convert labelme or detectron2 annotations to smaller size,
and crop images to animal faces only

## Our Team
