# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:16:04 2021
Updated on Wed Jan 26 2022: allowing alpha to be fix parameter
Last updated Wed Feb 22 2023: style, shortened for particular experiment

@author: abrielmann

Helper functions for fitting the data of the pilot experiment run in July 2021
"""
# import os, sys
import numpy as np
import simExperiment


def unpackParameters(parameters, n_features=2,
                     n_base_stims=7, n_morphs=5, fixParameters=None,
                     scaleVariances=False,
                     stimSpacing='linear'):
    """
    Read in a list with all parameter values in sorted order and returns them
    as named model parameters for passing them to further functions

    Parameters
    ----------
    parameters : list
        Sorted list of parameter values.
    n_features : int, optional
        Number of dimensions of the feature space. The default is 2.
    n_base_stims : int, optional
        Number of unique stimuli that are the source for creating the final,
        morphed stimulus space. The default is 7.
    n_morphs : int, optional
        Number of stimuli per morphed pair. The default is 5.
    fixParameters : dict, optional
        Dictionary containing values for any fixed model parameters.
        The dict may contain one or several of 'alpha', 'w_V', 'w_r', 'bias',
        'muState', 'varState', 'muTrue', 'varTrue', or 'features'.
        When fitting features for one of the stimuli while others
        are provided in 'features', the dict should also contain 'numStimsFit'.
        NOTE that the implementation of fitting one out of several stimuli at
        the moment only allows for fitting the first stimulus in the sequence.
        Consequences for morphed stimuli in feature space are taken into con-
        sideration
        The default is None.
    scaleVariances: bool, optional
        If true, variances will be adjusted such that variances increase
        with increasing feature dimension.
    stimSpacing : str, optional
        Either 'linear' or 'quadratic'. Determines how the morphs are spaced
        in between source images. 'quadratic' uses np.logspace to create 0.33
        and 0.67 morphs that are closer to the source images than 0.5 morphs.
        The default is 'linear'.

    Returns
    -------
    alpha : float
        Value for the learning rate parameter alpha.
    w_V : float
        Value for the weight of delta-V.
    w_r : float
        Value for the weight of immediate reward r.
    bias : float
        Value for the added bias parameter w_0.
    mu_0 : list of int
        List f length n_features containing means of the system state.
    cov_state : array_like
        Covariance matrix of the agent's system state, assumed to be spherical.
    mu_true : list of int
        List f length n_features containing means of the
        expected true distribution.
    cov_true : array_like
        Covariance matrix of the agent's expected true districution,
        assumed to be spherical.
    stims : array_like
        Array of n_features-dimensional representations of each stimulus.

    """

    paramsUsed = 4
    # basic assumption is that we fit all 4 structural parameters

    # get structural model parameters
    # for remaining parameters, check if any is set
    if fixParameters:
        if 'alpha' in fixParameters:
            alpha = fixParameters['alpha']
            paramsUsed = paramsUsed - 1
        else:
            alpha = parameters[0]
        if 'w_V' in fixParameters:
            w_V = fixParameters['w_V']
            paramsUsed = paramsUsed - 1
        else:
            w_V = parameters[paramsUsed-3]
        if 'w_r' in fixParameters:
            w_r = fixParameters['w_r']
            paramsUsed = paramsUsed - 1
        else:
            w_r = parameters[paramsUsed-2]
        if 'bias' in fixParameters:
            bias = fixParameters['bias']
            paramsUsed = paramsUsed - 1
        else:
            bias = parameters[paramsUsed-1]
    else:
        alpha = parameters[0]
        w_V = parameters[1]
        w_r = parameters[2]
        bias = parameters[3]

    # agent
    if fixParameters and 'muState' in fixParameters:
        mu_0 = fixParameters['muState']
    else:
        mu_0 = parameters[paramsUsed:paramsUsed+n_features]
        paramsUsed += n_features

    if fixParameters and 'varState' in fixParameters:
        cov_state = np.eye(n_features)*fixParameters['varState']
    else:
        cov_state = np.eye(n_features)*parameters[paramsUsed]
        paramsUsed += 1

    if fixParameters and 'muTrue' in fixParameters:
        mu_true = fixParameters['muTrue']
    else:
        mu_true = parameters[paramsUsed:paramsUsed+n_features]
        paramsUsed += n_features

    if fixParameters and 'varTrue' in fixParameters:
        cov_true = np.eye(n_features)*fixParameters['varTrue']
    else:
        cov_true = np.eye(n_features)*parameters[paramsUsed]
        paramsUsed += 1

    if scaleVariances:
        cov_state[range(n_features), range(n_features)] = [cov_state[0, 0]*ii for ii in range(1, n_features + 1)]

    # stimuli
    if fixParameters and 'features' in fixParameters:
        if fixParameters['numStimsFit'] == 1:
            start = paramsUsed
            base_stims = [parameters[start:start+n_features]]
            base_stims.extend(fixParameters['features'][1:n_base_stims])

        elif fixParameters['numStimsFit'] > 1:
            ValueError('Can only fit features for one'
                       + 'stimulus if others are fixed.')

        else:
            stims = fixParameters['features']
    else:
        base_stims = []
        # number of parameters already assigned
        start = paramsUsed
        for b in range(n_base_stims):
            base_stims.append(parameters[start:start+n_features])
            start += n_features

    if 'base_stims' in locals():
        stims = np.array(base_stims)
        pairs = [[0, 2], [0, 4], [0, 5], [0, 6],
                 [1, 2], [1, 3], [1, 4], [1, 5], [1, 6],
                 [4, 2], [4, 3],
                 [5, 3], [5, 4],
                 [6, 3], [6, 4], [6, 5]]

        for stimPair in pairs:
            s1 = base_stims[stimPair[0]]
            s2 = base_stims[stimPair[1]]
            if stimSpacing == 'linear':
                add_stims = np.linspace(s1, s2, n_morphs)
            elif stimSpacing == 'quadratic':
                add_left = np.geomspace(s1, s2, n_morphs)[:3]
                add_right = np.geomspace(s2, s1, n_morphs)[3:]
                add_stims = np.concatenate((add_left, add_right))
            else:
                raise ValueError(("Unknown stimulus spacing."
                                  + "Must be one of: linear; quadratic"))
            add_stims = add_stims[1:-1]  # don't keep the base images again
            stims = np.concatenate((stims, add_stims))

    return alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true, stims


def predict(parameters, data, n_features=2, n_base_stims=7, n_morphs=5,
            pre_expose=False, exposure_data=None, logTime=False,
            fixParameters=None, scaleVariances=False,
            stimSpacing='linear', replace_nan_rt=False,
            predict_trial_start=False):
    """
    Predict ratings for all stimuli as indexed by data.imageInd given their
    viewing time recorded as data.rt

    Parameters
    ----------
    parameters : list of float
        Sorted list of parameter values.
    data : pandas df
        pd.DataFrame containing the data to be predicted. Needs to contain the
        following columns: 'rt', 'imageInd'. The column imageInd needs
        to map onto the order in which stimuli are provided based on
        parameters.
    n_features : int, optional
        Number of dimensions of the feature space. The default is 2.
    n_base_stims : int, optional
        Number of unique stimuli that are the source for creating the final,
        morphed stimulus space. The default is 7.
    n_morphs : int, optional
        Number of stimuli per morphed pair. The default is 5.
    fixParameters : dict, optional
        Dictionary containing values for any fixed model parameters.
        The dict may contain one or several of 'w_V', 'w_r', 'bias', 'muState',
        'varState', 'muTrue', 'varTrue', or 'features'.
        When fitting features for one of the stimuli while others
        are provided in 'features', the dict should also contain 'numStimsFit'.
        NOTE that the implementation of fitting one out of several stimuli at
        the moment only allows for fitting the first stimulus in the sequence.
        Consequences for morphed stimuli in feature space are taken into con-
        sideration
        The default is None.
    pre_expose : boolean, optional
        Whether or not the agent is exposed to exposure_stims before start
        of predictions. The default is False.
    exposure_data : pandas df, optional
        pd.DataFrame containing data from the free viewing phase. Needs to
        contain image indices and viewing times. Only required if
        pre_expose=True.
        The default is None.
    logTime: boolean, optional.
        Whether or not to use the natural logarithm of the recorded response
        and viewing times as input for the model.
        The default is False.
    stimSpacing : str, optional
        Either 'linear' or 'quadratic'. Determines how the morphs are spaced
        in between source images. 'quadratic' uses np.logspace to create 0.33
        and 0.67 morphs that are closer to the source images than 0.5 morphs.
        The default is 'linear'.
    replace_nan_rt : bool, optional
        Whether or not to replace RTs that are nan with median RT.
        The default is False.
    predict_trial_start : bool, optional
        whether to predict A(t) at the beginning of the trial, i.e.,
        before updating mu. The default is False.

    Returns
    -------
    predictions : list of float
        Ratings as predicted by the model specified by parameters.

    """
    if not fixParameters:
        (alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true,
         raw_stims) = unpackParameters(parameters, n_features, n_base_stims,
                                       n_morphs, stimSpacing=stimSpacing,
                                       scaleVariances=scaleVariances)
    else:
        (alpha, w_V, w_r, bias, mu_0, cov_state, mu_true, cov_true,
         raw_stims) = unpackParameters(parameters, n_features, n_base_stims,
                                       n_morphs, fixParameters,
                                       stimSpacing=stimSpacing,
                                       scaleVariances=scaleVariances)

    if pre_expose:
        exposure_stims = raw_stims[exposure_data.imageInd.values, :]
        if logTime:
            exposure_stim_durs = np.round(np.log(exposure_data.viewTime.values))
        else:
            exposure_stim_durs = np.round(exposure_data.viewTime.values)
        for ii in range(len(exposure_stim_durs)):
            stim = exposure_stims[ii, :]
            dur = exposure_stim_durs[ii]
            # since the attention check introduces NaN values, it's important
            # to check for these and NOT include them
            if not np.isnan(dur):
                mu_0 = simExperiment.simulate_practice_trials(mu_0, cov_state,
                                                              alpha, stim,
                                                              1, dur)

    stims = raw_stims[data.imageInd.values, :]
    if logTime:
        stim_dur = np.round(np.log(data.rt.values))
    else:
        stim_dur = np.round(data.rt.values)
    if replace_nan_rt:
        # deal with nans in stimulus duration - replace with median rt
        if logTime:
            stim_dur[np.isnan(data.rt)] = np.log(data.rt).median()
        else:
            stim_dur[np.isnan(data.rt)] = data.rt.median()

    predictions = simExperiment.predict_ratings(mu_0, cov_state, mu_true,
                                                cov_true, alpha, w_r, w_V,
                                                stims, stim_dur, bias,
                                                predict_trial_start=predict_trial_start)
    return predictions
