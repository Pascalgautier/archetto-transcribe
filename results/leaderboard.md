| rank | engine | note onset F1 | onset+offset F1 | note F1 (no offset) | pitch acc. | fragmentation ↓ |
|---|---|---|---|---|---|---|
| 1 | ours | 0.994 | 0.964 | 0.992 | 0.998 | 0.006 |
| 2 | mt3 | 0.913 | 0.750 | 0.862 | 0.938 | 0.027 |
| 3 | crepe | 0.994 | 0.661 | 0.994 | 1.000 | 0.006 |
| 4 | pyin | 0.960 | 0.641 | 0.960 | 1.000 | 0.009 |
| 5 | basicpitch | 0.804 | 0.291 | 0.804 | 1.000 | 0.067 |

Per dataset:

| dataset | engine | onset F1 | onset+offset F1 | pitch acc. | fragmentation | est/ref notes | s |
|---|---|---|---|---|---|---|---|
| tango1_violon_I | basicpitch | 0.821 | 0.344 | 1.0 | 0.194 | 89/62 | 5.7 |
| tango1_violon_I | crepe | 1.0 | 0.984 | 1.0 | 0.0 | 62/62 | 50.9 |
| tango1_violon_I | mt3 | 0.976 | 0.646 | 0.968 | 0.081 | 65/62 | 26.7 |
| tango1_violon_I | ours | 1.0 | 1.0 | 1.0 | 0.0 | 62/62 | 53.5 |
| tango1_violon_I | pyin | 0.984 | 0.952 | 1.0 | 0.0 | 64/62 | 14.5 |
| vds_violon_I | basicpitch | 0.798 | 0.218 | 1.0 | 0.071 | 334/225 | 2.7 |
| vds_violon_I | crepe | 0.978 | 0.671 | 1.0 | 0.022 | 225/225 | 134.7 |
| vds_violon_I | mt3 | 0.911 | 0.741 | 0.925 | 0.018 | 212/225 | 37.4 |
| vds_violon_I | ours | 0.978 | 0.924 | 0.995 | 0.022 | 225/225 | 159.0 |
| vds_violon_I | pyin | 0.944 | 0.655 | 1.0 | 0.022 | 239/225 | 44.0 |
| vds_violon_II | basicpitch | 0.826 | 0.602 | 1.0 | 0.002 | 541/466 | 2.4 |
| vds_violon_II | crepe | 0.999 | 0.986 | 0.998 | 0.002 | 467/466 | 137.0 |
| vds_violon_II | mt3 | 0.844 | 0.7 | 0.86 | 0.009 | 468/466 | 57.3 |
| vds_violon_II | ours | 0.999 | 0.992 | 0.998 | 0.002 | 467/466 | 145.5 |
| vds_violon_II | pyin | 0.964 | 0.954 | 0.998 | 0.011 | 490/466 | 46.7 |
| vds_violon_III | basicpitch | 0.772 | 0.0 | 1.0 | 0.0 | 358/225 | 2.6 |
| vds_violon_III | crepe | 1.0 | 0.004 | 1.0 | 0.0 | 225/225 | 142.9 |
| vds_violon_III | mt3 | 0.923 | 0.915 | 1.0 | 0.0 | 258/225 | 39.5 |
| vds_violon_III | ours | 1.0 | 0.942 | 1.0 | 0.0 | 225/225 | 138.4 |
| vds_violon_III | pyin | 0.949 | 0.004 | 1.0 | 0.004 | 247/225 | 45.1 |
