# gllamar
alpaca motionlessly lies and eats some carrots instead of ryes [yet another hackathon project]

![StyleGAN2-generated camelids](https://github.com/Lyrical-Tokarev/gllamar/blob/main/assets/alpacas7_128_4x1.gif)

## WTF is this

Here you can find [ODS pet projects](https://ods.ai/projects/pet-projects) community's hackathon project.

The main goal: translate human photos to and from alpacas | llamas and other camelids. Or just have fun with alpacas. Because I love them.

Some or all of the code/interactive parts might be in Russian.

## Our Grand Plan

During the hackathon, which is held at October, 2 - October, 10, 2020 we plan to do:
- [x] collect alpaca | llama | other cutie photos
- [x] annotate them (we'll need bounding boxes with animal faces, since image-to-image translation will work better in this case)
- [ ] (this is done, but didn't went well) use magic of CycleGAN (most probably we'll use [this implementation](https://github.com/junyanz/CycleGAN))
- [+] ([trained stylegan](https://www.dropbox.com/s/f40bkbfsagziqxs/network-snapshot-000060.pkl), didn't blend yet) use StyleGAN2 to generate camelids, after that blend them with human faces!
- [?] (we have buggy bot, integration of cyclegan model is in process, stylegan model is not intergrated, and it might be very slow) box it: make Telegram bot, mobile application or just demo app with streamlit.

## Project limitations

- we might find not enough alpacas to achieve good results. For now we have only 432 images with camelids faces.
- tonz of other exciting projects with upcoming deadlines (like [this one](https://www.kaggle.com/c/stanford-covid-vaccine)) elsewhere, which are hard to resist, especially when you have 1-week long vacation and admire puzzles
- both of my teammates have no vacations now and have to do their work.

## Tools of trade we use

- pytorch
- detectron2
- CycleGAN
- [StyleGAN2 with shenanigans](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2)
- labelme for data annotation
- [pyTelegramBotPy](https://github.com/eternnoir/pyTelegramBotAPI)
- streamlit for visual debug and some screencasts

## Data: collection and preprocessing

We use OpenImages dataset, since it provides annotations (both segmentation and bounding boxes) for the class named 'Alpaca'.
Since annotations describe the whole animal bodies, and we've decided to use only animal faces, we've also used labelme to select bounding boxes for faces by hand.

For now we have 432 images with alpaca, llama or guanaco.

## Next steps and/or ideas:

- [x] add "labelme to detectron2" format converter
- [x] add script for detectron2 training with this preprocessed dataset and use it to make alpaca-detector
- [ ] ? add more images to our dataset from flickr and to extract animal faces with detectron2 trained with OpenImages-based Alpaca subset.
- [ ] validate collected and automatically annotated data
- [x] add script to convert labelme or detectron2 annotations to smaller size,
- [ ] add script to crop images to animal faces only based on detectron2 predictions
- [ ] publish dataset with cropped alpaca|llama|other camelid faces

## Our Team

- @latticetower (Tanya Malygina)
- @nmslana (Svetlana Dolgacheva)
- @ikwato (Anastasia Polevaya)
