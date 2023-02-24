# Code Review

## Notes

- I found no problems with the **correctness** of the code
  - Noemi would be a better judge of that, but I tried my best :-)
- So, most of the comments are about **understandability** and **style**
- Of those the understandability ones are the most important, whereas the style ones are things you may want to think about in the longer term
- And, of course, you are free to ignore any/all of what I said!

## General

- It is probably better to use `os.path.join(DIRNAME, FILENAME)` when creating paths, rather that `DIRNAME + "/" + FILENAME`, because it works better across OSs
  - There is actually a really nice python module named `pathlib` which no one uses, but which allows you to write e.g., `Path(DIRNAME) / FILENAME` as if you were just writing a path
- Also, using f-strings if often nice than `+`-ing strings together
- You seem to go back and forward between `camelCase` and `snake_case` naming. It might be better to choose one and stick with it
- Is it worth saying in the main comments for each file what part of publication it is implementing (or, rather, which part of the paper is explaining that file)?

## Specific file

### `README.md`

- Added section explaining how to get, install, and run the code
- Move the bit about requirements to that section
  - This can actually be encoded in the `requirement.txt` file...
- Changed the name of `vgg_features`. It was still using the old capitalisation
- Actually, after reviewing the analysis, I wonder if it might be worth explaining the flow of the analysis code
  - I mean, showing what to run, in what order, and what gets created as you do

### `figureFunctions.py`

- More comments?
  - If you got the code from somewhere, where was it from?
  - Is it worth commenting on what the arguments and returns are?
  - Or, expand the header comment to describe more about the file (since it is also not mentioned in the `README`)?

### `fitPilot.py`/`simExperiment.py`

- Great comments!
- I would also add a link, in the comments to the repo where the code is from
- Note: I have onlys skimmed these two, since the are from elsewhere, and were presumably checked previously...

### `plot_images_inVGGspace.py`

- From what I can see, the code here seems fine
- What is missing is an explanation of what the file is for, and how it is used:
  - Are users supposed to run it?
  - Is it called from another file?
  - etc.

### `analysis/a_get_complete_data.py`

- Lines 39+:
  - That is a lot of lists. Given that you package them into dataframes later, could you do it as a dictionary?
- Lines 83 - 85:
  - `ast` is a pretty unusual thing to be using. Is it worth making a note of what it is doing?
- Lines 144-165 and 226-246:
  - These two block are doing the same thing, right? Is it worth extracting them to a function?

### `analysis/b_get_descriptives_ratings.py`/`analysis/c_get_descriptives_participants.py`/others

- Both of these files use `participantList` (which I assume is the same in both cases)
  - And then it has to be sorted...
- Could you put the participants in a separate file, and read it in, rather than hardcode it in a bunch of different places?
  - Or, put it in a python file, and just import it where you need it.

### `analysis/d_fit_custom_model_ratings_cv.py`

- Lines 36, 37:
  - Using `''` as the zero value here (for e.g., `truncatedPredictions`) is a bit strange for me
  - I realise that you will be using them as strings later, but I think at least some of them are also used in conditionals
  - It is possible to set them to either None or a string and do `truncatedPredictions if truncatedPredictions else ''` when you want to turn it into a string later?
- Lines 62+:
  - Maybe a bit of white space in between the unrelated if statements to show that they are unrelated
- Line 134: Not really sure what `pairs` is for 
- Line 215: `Applis` => `Applies`
- Line 230:
  - I realise this is actually a comment about `unpackParameters`, but with that many return values, it might be worth returning a dictionary/object, rather than having to ignore so many values
- Lines 300+: you use `data.iloc[:55]` a lot. Is it worth assigning that value to something, and giving it a name? It seems a bit magical otherwise
- Line 318: Is `rmseFitList` being used?
- Line 359: It took me a while to work out that this save was per participant, whilst the next one was overall. Maybe mention in the comments what you are saving

### `analysis/e_evaluate_costum_model_fits.py`

- See comments above re: `''` as a zero value
- See comments above re: lots of lists vs. dictionaries

### `analysis/f_merge_fit_results_participantInfo_cv.py`

- I'm not 100% sure what this file is doing
  - I understand that it is merging other files, but it is also doing some value munging...
- Commenting?

### `analysis/g_get_leaveOneOut_Average_predictions.py`

- Great, no complaints :-)

### `analysis/h_compare_models_cv.py`

- Code is fine, but a few more comments would be nice
  - Even if it is just a module comment that gives an overview of what the module is for

### `analysis/i_sim_scrambled_trial_order_with_refit.py`

- Lines 6, 7, 8: Maybe a comment for each explaining what these lists are for
- Line 87: `featuers` => `features`
- Lines 123, 130: Are these the same `pred_ratings` and `cost_fn` functions from `analysis/d_fit_custom_model_ratings_cv.py`
  - If so, maybe extract somewhere?
- Line 153: Why `participantList[:1]`?
- Lines 164+: There is the magic `[:55]` again...
- Lines 227+: Is there a reason for the `+=`, rather than using `.append`?

### `analysis/j_eval_sim_scrambled_trial_order_with_refit.py`

- Lines 49 - 130:
  - This code is the same as is in `i_sim_scrambled_trial_order_with_refit`, right?
  - Can it be extracted somewhere?
  - Actually is it even being used in this file? I am not sure that it is, and if not perhaps it can be removed.
- Lines 150+: There is the magic `[:55]` again...
- Lines 167+: Is there a reason for the `+=`, rather than using `.append`?

### `analysis/k_model_selection_cv.py`

- Lines 56, 57: `spher` and `normal` seem to be unused
- Lines 67, 100: Is there a reason why you are printing this to latex?
- Again, perhaps a module comment to explain what this is doing and why.
