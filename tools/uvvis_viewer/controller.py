from a304data.uvvisdataset import UVVisDataset


class UVVisController:
    def load_dataset(self, folder):
        ds = UVVisDataset(folder)
        ds.calculate_absorbance()
        return ds

    def apply_find_peak(self, ds, height, prominence, distance):
        ds.find_peak(
            height=height if height > 0 else None,
            prominence=prominence if prominence > 0 else None,
            distance=distance if distance > 0 else None
        )

    def plot(self, ds, index, ax, wl_min, wl_max):
        data = ds.uvvis_data[index]

        ax.plot(
            ds.bg_data['Wavelength'],
            data['Absorbance']
        )

        # 标峰
        if hasattr(ds, "peaks"):
            for peak_idx in ds.peaks:
                ax.axvline(
                    data['Wavelength'].iloc[peak_idx],
                    color='k',
                    linestyle='--',
                    alpha=0.6
                )
                ax.text(
                    data['Wavelength'].iloc[peak_idx],
                    1.02,
                    f"{data['Wavelength'].iloc[peak_idx]:.1f}",
                    rotation=90,
                    fontsize=9
                )

        ax.set_xlim(wl_min, wl_max)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Absorbance")
        ax.set_title("UV-Vis Spectrum")
