# README

## Overview

This folder contains essential resources used in the evaluation of disentanglement for our project. Inside, you will find two key files:

1. **White Image File**: This image serves as the base from which all latent representations are computed. The latent representations are generated in response to various prompts that are systematically varied according to the attributes and objects described in the accompanying JSON file.

2. **JSON File**: This file is a crucial component of our evaluation process. It outlines all the possible combinations of objects and attributes that are integral to our disentanglement analysis.

## Contents

### JSON File Structure

The JSON file includes a comprehensive list of objects and attributes:

- **Objects**: A collection of items such as 'chair', 'table', etc.
- **Attributes**: These are categorized into several types, each with its own list of values. For example:
  - **Colors**: Includes values like 'red', 'green', etc.
  - **Textures**: Encompasses a variety of textures such as 'steel', 'wood', etc.

### Purpose

The primary objective of these resources is to facilitate the generation of a diverse range of attribute-object combinations. These combinations are then used to produce latent representations for thorough evaluation of disentanglement in our system. The process ensures a comprehensive assessment by covering a broad spectrum of scenarios and possibilities.