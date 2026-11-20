Project copies of Raphael's settings go here, with this project's NWB
metadata filled in (lab, experimenters, procedures, keywords):

    cp <Raphael>/src/raphael/configs/ephys.yaml configs/
    cp <Raphael>/src/raphael/configs/imaging.yaml configs/

and are passed to processing jobs with `--config configs/ephys.yaml`.
A `configs/cluster.yaml` here replaces Raphael's cluster site settings.
