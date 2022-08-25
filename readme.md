# Codes and demo for ICASSP 2022 submission. 

This repo includes two models: a harmonization model and a melody generation model. Both models apply adversarial intervention to disentangle representation from sequential conditions to gain better control. Due to limited space, we only present the harmonization model in the paper. Our paper is under single-blind review at the moment.

## Demo of Harmonization
See `./harmonization_model_ver8/write`.

## Demo of Melody Generation
See `./melody_generation_model/write/melody_write`.

## Contact
Jingwei Zhao (PhD student in Data Science at NUS)

jzhao@u.nus.edu

Oct. 07, 2021


<midi-player src="./demo/control_2_modal_change.mid" sound-font>
</midi-player>

<!-- The following needs to be inserted somewhere on the page for the player(s) to work. -->
<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.22.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>