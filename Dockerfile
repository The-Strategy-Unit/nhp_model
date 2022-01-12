FROM rocker/r-rspm:20.04

RUN install.r remotes
# copy just the description file to speed up rebuilding the image, then install the required packages
COPY nhpmodel/DESCRIPTION /opt/nhpmodel/
RUN Rscript -e 'remotes::install_deps("/opt/nhpmodel", dependencies = TRUE)'

VOLUME [ "/mnt/data", "/mnt/queue", "/mnt/results" ]

# handle the rest of the files now and install the package
COPY nhpmodel/ /opt/nhpmodel
RUN R CMD INSTALL /opt/nhpmodel

COPY run_models.R /opt

CMD ["/usr/bin/Rscript", "/opt/run_models.R"]