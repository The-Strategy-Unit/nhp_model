FROM rocker/r-rspm:20.04

RUN install.r remotes
# copy just the description file to speed up rebuilding the image, then install the required packages
COPY nhpmodel/DESCRIPTION /opt/nhpmodel/
RUN Rscript -e 'remotes::install_deps("/opt/nhpmodel", dependencies = TRUE)'

# handle the rest of the files now and install the package
COPY nhpmodel /opt/
RUN R CMD INSTALL /opt/nhpmodel

VOLUME [ "/mnt/data", "mnt/results" ]

CMD ["/bin/bash"]