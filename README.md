## Background
This project was motivated by the desire to expand open innovation and by responses offered to recomendations in this [report](https://www.cdc.gov/niosh/topics/surveillance/pdfs/A-Smarter-National-Surveillance-System-Final.pdf). The Centers for Disease Control and Prevention (CDC) used crowdsourcing to improve a Natural Language Processing (NLP) Machine Learning (ML) algorithm to code unstructured work-related injury narratives to [OIICS](https://wwwn.cdc.gov/wisards/oiics/Trees/MultiTree.aspx?Year=2012) codes. An intra- and extra-mural competition was held in 2019. These algorithms are the top-five performing NLP solutions created by the crowdsourcing competitions. The project was made prossible by funding from [NIOSH-DSR](https://www.cdc.gov/niosh/contact/im-dsr.html) and the [CDC-OS-OTI](https://www.cdc.gov/os/technology/index.htm). A multitute of scientific workgroups at the CDC *significantly* contributed towards the promotion of the project and the recruitment of intra-mural competitors. Our effort was promoted by [FCPCCS](https://www.citizenscience.gov/about/community-of-practice/#) and the [AIC](https://digital.gov/communities/artificial-intelligence/). Our team has participated with multiple outlets on interviews and was invited by MIT's [J-Clinic](https://www.jclinic.mit.edu/) and Harvard's [LISH](https://lish.harvard.edu/) to speak about innovative project. Our team, the **//m_BrainGineers**, is deeply grateful to the funding parties and all those who dedicated their time to helping achieve this amazing success. If you have any questions or comments, please send an [email](nej6@cdc.gov).    

## //m_BrainGineers
Our team of 16 federal employees from 7 federal agencies was led by [Dr. Siordia](https://www.linkedin.com/in/carlos-siordia-phd-03b152b9/) and [Dr. Bertke](https://www.linkedin.com/in/steve-bertke-3bb49a9a/) in close collaboration with [Mr. Measure](https://www.linkedin.com/in/ameasure) and [Dr. Russ](https://www.linkedin.com/in/daniel-russ-9541aa15/). Federal agencies partipating included the [CDC](https://www.cdc.gov/), [BLS](https://www.bls.gov/), [NIH](https://www.nih.gov/), [CENSUS](https://www.census.gov/), [CPSC](https://www.cpsc.gov/), [FEMA](https://www.fema.gov/), and the [OSHA](https://www.osha.gov/). All team members contributed and are listed by including their host federal agency and center/institute/office(CIO):
 - Carlos Siordia PhD (CDC-NIOSH)  
 - Steve Bertke PhD (NIOSH-NIOSH) 
 - Audrey Reichard MPH (CDC-NIOSH) 
 - Syd Webb PhD (CDC-NIOSH)
 - Alexander Measure MS (BLS-OSHS) 
 - Daniel Russ PhD (NIH-CIT) 
 - Stacey Marovich MHI (CDC-NIOSH) 
 - Kelly Vanoli BS (CDC-NIOSH)
 - Mick Ballesteros PhD (CDC-NCIPC)
 - Jeff Purdin MS (CDC-NIOSH)
 - Melissa Friesen PhD (NIH-NCI) 
 - Machell Town PhD (CDC-NCCDPHP) 
 - Lynda Laughlin PhD (CENSUS-SEHSD)
 - Tom Schroeder MS (CPSC-DHIDS)  
 - Jim Heeschen MS (FEMA-USFA-NFDC)
 - Miriam Schoenbaum PhD (OSHA-OSA) 
 - Ari Miniño MPH (CDC-NCHS) 

## Results of Intramural (within CDC) Competition
The CDC has been using machine learning for decades as discussed in this [blog](https://blogs.cdc.gov/genomics/2020/09/17/artificial-intelligence/). Our project represents the first time the CDC hosted an intramural NLP marathon. The 19 intra-mural competitors are each extraordinary analyst! For our competition, they had the courage to put their reputation on the line. They procured management approval for participation and devoted many hours to developing their solution. Even those new to NLP performed admirably. We are grateful for their contributions. More information on 19 competitors in 9 teams is provided in this [announcement](https://www.cdc.gov/niosh/updates/upd-10-24-19.html) and this [blog](https://blogs.cdc.gov/niosh-science-blog/2020/02/26/ai-crowdsourcing/). Competitors had to code unstructured work-related injury narratives to 48 unique [OIICS](https://wwwn.cdc.gov/wisards/oiics/Trees/MultiTree.aspx?Year=2012) two-digit "event" codes. Competitors has access to a [NEISS-Work](https://wwwn.cdc.gov/wisards/workrisqs/) data files from 2012 through 2016 with 191,835 observations. Results are listed by ranking (including weighted F1 score):
1. (wF1=87.77) **Scott Lee PhD** in CDC's Center for Surveillance, Epidemiology, and Laboratory Services (CSELS) used an ensemble classifier blending four BERT neural network models.
2. (wF1=87.15) **Mohammed Khan MS and Bill Thompson PhD** from CDC-NCHHSTP's Division of Viral Hepatitis (DVH) used Recurrent Neural Network with Fastai on Codalab.  
3. (wF1=84.47) **Jasmine Varghese MS, Benjamin Metcalf MA, and Yuan Li PhD** from CDC-NCIRD's Division of Bacterial Diseases (DBD) used Regularized Logistic Regression with custom word corpus. 
4. (wF1=84.45) **Keming Yuan MS** from CDC-NCIPC's Division of Violence Prevention (DVP) used Long Short-Term Memory Recurrent Neural Network. 
5. (wF1=83.32) **Naveena Yanamala PhD** from CDC-NIOSH's Health Effects Laboratory Division (HELD) used Linear Support Vector Model post custom standardization.
6. (wF1=82.75) **Li Lu MD PhD** from CDC's Office of the Chief Operating Officer (OCOO) used an ensemble classifier using Regularized Logistic Regression, Multi-Layer Perceptron, and Linear Support Vector Models.
7. (wF1=81.47) **Joan Braithwaite MSPH** in CDC's National Center for Chronic Disease Prevention & Health Promotion (NCCDPHP) used Linear Support Vector Model post lemmatization. 
8. (wF1=81.00) **Donald Malec PhD** from CDC-NCHS' Division of Research and Methodology (DRM) used Support Vector Machine.
9. (wF1=77.45) **José Tomás Prieto PhD and Faisal Reza PhD** from CDC-CSELS' Division of Scientific Education & Professional Development (DSEPD) used Regularized Logistic Regression—Lasso.



## Results of Extramural (international) Competition
Using an Inter-Agency Agreement (IAA) between NASA's [CoECI](https://www.nasa.gov/offices/COECI/index.html) and [NIOSH](https://www.cdc.gov/niosh/contact/im-dsr.html), [TopCoder](https://www.topcoder.com/) was contracted and hosted an international [competition](https://www.topcoder.com/challenges/020c0f34-1f05-4d58-9530-680280a2994b) to develop natural language processing algorithms. 
We give special thanks to TopCoder's [Dr. Contreras](https://www.linkedin.com/in/michael-contreras-056873b/) and [Mr. Reitz](https://www.linkedin.com/in/danreitz1/).
International competition had 388 registrands from 26 countries. About 32% of competitors where from the USA and 21% from India. A total of 20 universities where represented in the competition. We are grateful for willingness to consider our challenge and invest the time required to outperform our "CDC model" (i.e., the model created by [Dr. Lee](https://www.linkedin.com/in/scott-lee-b767a1144/)). Competitors used [NEISS-Work](https://wwwn.cdc.gov/wisards/workrisqs/) data from 2012 through 2017, which included a total of 229,820 observations. Competitors had to use unstructured injury narratives to code "two-digit OIICS-event" codes. TopCoder had a total of 961 total unique submissions of algorithms to be scored and ranked. Here are the prize-winners, listed by ranking (including weighted F1 score):
1. (wF1=89.20) **Raymond van Venetië** from Netherlands used an ensemble classifer with ALBERT and DAN models. 
2. (wF1=89.12) **2nd Pavel Blinov** from Russia used used an ensemble classifer with BERT, XLNet, and RoBERTa models.
3. (wF1=89.09) **Zhuoyu Wei** from China used an ensemble classifer that included RoBERTa models. 
4. (wF1=88.99) **Zhensheng Wang** from USA used an ensemble classifier that included XLNet and BERT models.
5. (wF1=88.93) **A Sai Sravan** from India used an ensemble classifier that included BERT and RoBERTa models. 

## Disclaimer 
Use of these algorithms and associated files does not imply a NASA or CDC endorsement of any one particular product, service or enterprise. U.S. Government logos, including the NASA and CDC logo, cannot be used without express, written permission. These algorithms were designed based on the best available science and should not be modified or altered. Use of these algorithms must be accompanied by the following disclaimer and non-endorsement language:

```This natural language processing (NLP) algorithm was originally developed through a NASA-CDC collaboration with the public community of programmers. Neither NASA nor CDC guarantee the accuracy of the NLP algorithm. The U.S. Government does not make any warranty of any kind, either expressed or implied, for any non-U.S. Government version of the NLP algorithm . Use by a non-U.S. Government organization or enterprise does not imply a U.S. Government endorsement of any one particular product, service, or enterprise or that this use of the NLP algorithm represents the official views of the U.S. Government.```

## Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication.](https://creativecommons.org/publicdomain/zero/1.0/) All contributions to this repository will be released under the CC0 dedication. By submitting a pull request you are agreeing to comply with this waiver of copyright interest.

## License Standard Notice

This project constitutes a work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105. The project utilizes code licensed under the terms of the Apache Software License and therefore is licensed under ASL v2 or later. This program is free software: you can redistribute it and/or modify it under the terms of the Apache Software License version 2, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache Software License for more details. You should have received a copy of the Apache Software License along with this program. If not, see this [license](http://www.apache.org/licenses/LICENSE-2.0.html).

## Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and
information. All material and community participation is covered by the
[Disclaimer](https://github.com/CDCgov/template/blob/master/DISCLAIMER.md)
and [Code of Conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md).
For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

## Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) and submitting a pull request. (If you are new to GitHub, you might start with a [basic tutorial](https://help.github.com/en/github/getting-started-with-github/set-up-git)). By contributing to this project, you grant a world-wide, royalty-free, perpetual, irrevocable, non-exclusive, transferable license to all users under the terms of the [Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

All comments, messages, pull requests, and other submissions received through CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act and may be archived. Learn more [here](http://www.cdc.gov/other/privacy.html).

## Records Management Standard Notice

This repository is not a source of government records, but is a copy to increase collaboration and collaborative potential. All government records will be published through the [CDC web site.](http://www.cdc.gov)

## Additional Standard Notices

Please refer to [CDC's Template Repository](https://github.com/CDCgov/template/blob/master/open_practices.md) for more information about [contributing to this repository, public domain notices and disclaimers](https://github.com/CDCgov/template/blob/master/open_practices.md), and [code of conduct](https://github.com/CDCgov/template/blob/master/code-of-conduct.md). Learn more about CDC GitHub Practices for Open Source Projects [here](https://github.com/CDCgov/template/blob/master/open_practices.md).

# General Disclaimer 
This repository was created for use by NASA and CDC programs to collaborate on public health related projects in support of agency mission. Github is not hosted by the NASA or the CDC, but is a third party website used by NASA and CDC and its partners to share information and collaborate on software. Neither NASA's nor CDC’s use of GitHub imply an endorsement of any one particular service, product, or enterprise.
