import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit import Minuit, minimize
from scipy import integrate
from scipy import stats
import pandas as pd
from scipy.optimize import curve_fit
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource,Title


# Define the layout of the app
st.title("Muon Analysis App")

# File upload section
st.sidebar.title("Data Import")
file = st.sidebar.file_uploader("Upload the mc.bin file", type="bin")

# Graph selection section
st.sidebar.title("Graph Selection")
graph_options = ['Data','Invariant Mass', 'Transverse Momentum', 'Pseudorapidity','Total momentum','Transverse momentum of first muon','Transverse momentum of second muon','Best curve fitting for thr first peak','background estimation for the Muon Pair','Full fit']
selected_graph = st.sidebar.radio("Select a graph", graph_options)




# Import data and create the pandas data frame
if file is not None:

    # Slider for controlling the number of muons
    data = file.read()
    datalist = np.frombuffer(data, dtype=np.float32)
    nevent = int(len(datalist) / 6)
    xdata = np.split(datalist, nevent)
    cols = ["InvMass", "TransMomPair", "PseudoRapid", "MomPair", "TransMom1", "TransMom2"]
    df = pd.DataFrame(xdata, columns=cols)
    # Display the DataFrame on a separate page


    num_muons = st.slider("Number of Muons", min_value=1, max_value=df.shape[0], value=10, step=10)



    df=df.head(num_muons)
    

    if selected_graph == 'Data':
        if st.sidebar.radio("Show DataFrame", [False, True]):
            st.title("Data Frame")
            st.dataframe(df)
            st.write("Number of Enteries:", df.shape[0])
    
    
    elif selected_graph == 'Invariant Mass':
        st.sidebar.title("Interactivity")
        interactiveness_options={"Interactive", "Non-Interactive"}
        selected_option= st.sidebar.radio("Select an option", interactiveness_options)
        # Plot histogram of the data and save it into a .png
        if selected_option == 'Non-Interactive':
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(df['InvMass'], bins=1000, color='thistle', density=True)
            ax.set_xlabel('Mass (GeV/c^2)')
            ax.set_ylabel('Candidates')
            ax.set_title('Muon Pair Invariant Mass')

            st.pyplot(fig)
        elif selected_option == 'Interactive':
            p = figure(title='Muon Pair Invariant Mass', x_axis_label='Mass (GeV/c^2)', y_axis_label='Candidates')
            hist, edges = np.histogram(df['InvMass'], bins=1000, density=True)
            source = ColumnDataSource(data=dict(hist=hist, left=edges[:-1], right=edges[1:]))
            p.quad(top='hist', bottom=0, left='left', right='right', source=source, color='thistle')

            # Add hover tool to display coordinates
            hover = HoverTool(tooltips=[('Mass', '@left{0.00} - @right{0.00} GeV/c^2'), ('Candidates', '@hist')])
            p.add_tools(hover)

            # Create an interactive plot using Streamlit
            st.bokeh_chart(p)

        
    elif selected_graph == 'Transverse Momentum':
        # Cut the data depending on Transverse Momentum
        df = df[(df['TransMomPair'] >= 0) & (df['TransMomPair'] <= 15)]

        # Sideband subtraction for Transverse Momentum
        peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
        s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
        s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

        # Create a 2D plot of Transverse Momentum against Invariant Mass()
        fig1,ax1=plt.subplots()
        hist2d, xedges, yedges, im = ax1.hist2d(peak1['InvMass'], peak1['TransMomPair'], bins=100, cmap=plt.cm.CMRmap)
        ax1.set_ylabel("Transverse Momentum of Muon Pair [GeV/c]")
        ax1.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
        plt.colorbar(im,ax=ax1)
        st.pyplot(fig1)

        fig2,ax2=plt.subplots()
        obs = ax2.hist(peak1['TransMomPair'], bins=20, alpha=0.5,label="obs")
        bck1 = ax2.hist(s1['TransMomPair'], bins=20, alpha=0.5,label="bck1")
        bck2 = ax2.hist(s2['TransMomPair'], bins=20, alpha=0.5,label="bck2")

        n = obs[0] - bck1[0] - bck2[0]

        # Plot signal, background, and the original distribution as a function of transverse momentum
        ax2.plot(obs[1][1:], n, 'bx', label='signal')
        ax2.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label='Bck')
        ax2.plot(obs[1][1:], obs[0], 'gx', label='Data')
        ax2.set_xlabel('Muon Pair Transverse Momentum (GeV/c)')
        ax2.set_ylabel('Candidates')
        ax2.legend()
        st.pyplot(fig2)

    elif selected_graph == 'Pseudorapidity':
        # Cut the data depending on Pseudorapidity
        df = df[(df['PseudoRapid'] >= 2) & (df['PseudoRapid'] <= 6)]

        # Sideband subtraction for Pseudorapidity
        peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
        s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
        s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

        # Create a 2D plot of Pseudorapidity against Invariant Mass
        fig3,ax3=plt.subplots()
        hist2d, xedges, yedges, im=ax3.hist2d(peak1['InvMass'], peak1['PseudoRapid'], bins=100, cmap=plt.cm.CMRmap)
        ax3.set_ylabel("Pseudorapidity of Muon Pair")
        ax3.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
        plt.colorbar(im,ax=ax3)
        st.pyplot(fig3)
        fig4,ax4=plt.subplots()
        obs = ax4.hist(peak1['PseudoRapid'], bins=20, alpha=0.5)
        bck1 = ax4.hist(s1['PseudoRapid'], bins=20, alpha=0.5)
        bck2 = ax4.hist(s2['PseudoRapid'], bins=20, alpha=0.5)

        n = obs[0] - bck1[0] - bck2[0]

        # Plot signal, background, and the original distribution as a function of pseudorapidity
        ax4.plot(obs[1][1:], n, 'bx', label='signal')
        ax4.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label='Bck')
        ax4.plot(obs[1][1:], obs[0], 'gx', label='Data')
        ax4.set_xlabel('Pseudorapidity (GeV/c)')
        ax4.set_ylabel('Candidates')
        ax4.legend()
        st.pyplot(fig4)

    elif selected_graph == 'Total momentum':
        # Identical process as above, however for the total momentum of the muon pair
        df = df[(df['MomPair'] >= 0) & (df['MomPair'] <= 300)]

        peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
        s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
        s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

        fig5,ax5=plt.subplots()

        hist2d, xedges, yedges, im=ax5.hist2d(peak1['InvMass'], peak1['MomPair'], bins=100, cmap=plt.cm.CMRmap)
        ax5.set_ylabel("(Total) Momentum of Muon Pair [GeV/c]")
        ax5.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
        plt.colorbar(im,ax=ax5)
        st.pyplot(fig5)

        fig6,ax6=plt.subplots()
        obs = ax6.hist(peak1['MomPair'], bins=20, alpha=0.5,label="obs")
        bck1 = ax6.hist(s1['MomPair'], bins=20, alpha=0.5,label="bck1")
        bck2 = ax6.hist(s2['MomPair'], bins=20, alpha=0.5,label="bck2")

        n = obs[0] - bck1[0] - bck2[0]

        ax6.plot(obs[1][1:], n, 'bx', label='signal')
        ax6.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label='Bck')
        ax6.plot(obs[1][1:], obs[0], 'gx', label='Data')
        ax6.set_xlabel('(Total) Momentum of Muon Pair (GeV/c)')
        ax6.set_ylabel('Candidates')
        ax6.legend()
        st.pyplot(plt)

    elif selected_graph == 'Transverse momentum of first muon':
         # Cut the data depending on the transverse momentum of the first muon
        df = df[(df['TransMom1'] >= 1) & (df['TransMom1'] <= 10)]

        # Sideband subtraction for the transverse momentum of the first muon
        peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
        s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
        s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]

        # Create a 2D plot of the transverse momentum of the first muon against invariant mass
        fig1,ax1=plt.subplots()
        hist2d, xedges, yedges, im = ax1.hist2d(peak1['InvMass'], peak1['TransMom1'], bins=100, cmap=plt.cm.CMRmap)
        ax1.set_ylabel("Transverse Momentum of First Muon [GeV/c]")
        ax1.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
        plt.colorbar(im,ax=ax1)
        st.pyplot(plt)

        fig2,ax2=plt.subplots()
        obs = ax2.hist(peak1['TransMom1'], bins=20, alpha=0.5,label="obs")
        bck1 = ax2.hist(s1['TransMom1'], bins=20, alpha=0.5,label="bck1")
        bck2 = ax2.hist(s2['TransMom1'], bins=20, alpha=0.5,label="bck2")

        n = obs[0] - bck1[0] - bck2[0]

        ax2.plot(obs[1][1:], n, 'bx', label='signal')
        ax2.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label= 'Bck')
        ax2.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
        ax2.set_xlabel('Transverse Momentum of First Muon [GeV/c]')
        ax2.set_ylabel('Candidates')
        ax2.legend()
        st.pyplot(plt)

    elif selected_graph == 'Transverse momentum of second muon':
         # Cut the data depending on the transverse momentum of the second muon
        df = df[(df['TransMom2'] >= 1) & (df['TransMom2'] <= 10)]

        # Sideband subtraction for the transverse momentum of the second muon
        peak1 = df[(df['InvMass'] >= 9.3) & (df['InvMass'] <= 9.6)]
        s1 = df[(df['InvMass'] >= 9.15) & (df['InvMass'] <= 9.3)]
        s2 = df[(df['InvMass'] >= 9.6) & (df['InvMass'] <= 9.75)]
        fig1,ax1=plt.subplots()
        # Create a 2D plot of the transverse momentum of the second muon against invariant mass
        heatmap=ax1.hist2d(peak1['InvMass'], peak1['TransMom2'], bins=100, cmap=plt.cm.CMRmap)
        ax1.set_ylabel("Transverse Momentum of Second Muon [GeV/c]")
        ax1.set_xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
        cbar = fig1.colorbar(heatmap[3], ax=ax1)
        st.pyplot(fig1)
        fig2,ax2=plt.subplots()


        obs = ax2.hist(peak1['TransMom2'], bins=20, alpha=0.5)
        bck1 = ax2.hist(s1['TransMom2'], bins=20, alpha=0.5)
        bck2 = ax2.hist(s2['TransMom2'], bins=20, alpha=0.5)

        n = obs[0] - bck1[0] - bck2[0]

        ax2.plot(obs[1][1:], n, 'bx', label='signal')
        ax2.plot(obs[1][1:], bck1[0] + bck2[0], 'rx', label= 'Bck')
        ax2.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
        ax2.set_xlabel('Transverse Momentum of Second Muon [GeV/c]')
        ax2.set_ylabel('Candidates')
        ax2.legend()
        st.pyplot(fig2)

    peak1 = df[ (df['InvMass']>=9) & (df['InvMass']<=9.7) ]
    def pdf_comb(x, F, mu, sigma, tau):
    
        """
        Probability distribution function (pdf) of a gaussian and exponential 
        """
        
        a = 9
        b = 9.7
        
        gaussian = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2 / (2*sigma**2)) 
        exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

        
        return F*(exponential) + (1-F)* ( gaussian )
    
    def likelihood(F, mu, sigma, tau):
    
        """
        Likelihood function of gaussian and exponential fit
        """
        
        return -np.sum(np.log(pdf_comb(peak1['InvMass'], F, mu, sigma, tau)))
        # Determine optimal parameters for fitting of the first peak
    m_comb = Minuit(likelihood, F=0.6, mu=9.45, sigma=0.3, tau=1.6)

    
    if selected_graph == 'Best curve fitting for thr first peak':

        st.dataframe(peak1)
        

   #Plot the mass histogram, fitted function, and signal distribution

        fig1,ax1=plt.subplots()

        n1, bins1, patches1 = ax1.hist(peak1['InvMass'], bins = 1000, color = 'thistle', density = True, label = 'Mass Histogram')
        st.write(bins1)
        y_fit = pdf_comb(bins1[:-1], *m_comb.values)

        st.pyplot(fig1)
        fig2,ax2=plt.subplots()

        ax2.plot(bins1[:-1], y_fit, color = "darkBlue", label = 'Fitted Function')
        ax2.set_xlabel('Mass (GeV/c^2)')
        ax2.set_ylabel('Candidates')
        ax2.set_title('Muon Pair Invariant Mass for Y(1S) peak')

        gaussian = (1-m_comb.values[0])*(1/(m_comb.values[2]*np.sqrt(2*np.pi))) * np.exp(-(bins1[:-1]-m_comb.values[1])**2 / (2*m_comb.values[2]**2))

        ax2.plot(bins1[:-1], gaussian, color = 'red', linestyle = '--', label = 'Signal Distribution')
        ax2.legend()
        st.pyplot(fig2)


        
     # Cut the data to isolate a region with just background
    bgn = df[(df['InvMass'] >= 10.5)]


    # Define the PDF for background exponential curve
    def pdf_bgn(x, tau):
        a = float(bgn['InvMass'].min())
        b = float(bgn['InvMass'].max())
        return (1 / tau) * np.exp(-x / tau) / (np.exp(-a / tau) - np.exp(-b / tau))
    


        # Define the likelihood function for background exponential fit
    def likelihood_bgn(tau):
        return -np.sum(np.log(pdf_bgn(np.array(bgn['InvMass']), tau)))
    

    m_comb_bgn = Minuit(likelihood_bgn, tau = 1.76)

    m_comb_bgn.migrad()  # run optimiser
    m_comb_bgn.hesse()   # run covariance estimator

    
    if selected_graph =="background estimation for the Muon Pair":
       

        

        # Fit the exponential curve to the background data
        result = minimize(likelihood_bgn, x0=[1.76])
        tau = result.x[0]

        # Create the background exponential curve
        x_vals = np.linspace(bgn['InvMass'].min(), bgn['InvMass'].max(), 100)
        y_vals = pdf_bgn(x_vals, tau)

        # Create a DataFrame for Bokeh plot
        bokeh_data = pd.DataFrame({'x': x_vals, 'y': y_vals})

        # Choose between Bokeh and Matplotlib plots using a radio button
        plot_option = st.sidebar.radio("Select a plot", ('Bokeh', 'Matplotlib'))

        # Plot the selected plot
        if plot_option == 'Bokeh':
            # Load data and perform fitting


            # Perform histogram
            n_bgn, bins_bgn, _ = plt.hist(bgn['InvMass'], bins=1000, color='thistle', density=True, label='Mass Histogram')

            # Perform histogram
            n_bgn, bins_bgn, _ = plt.hist(bgn['InvMass'], bins=1000, color='thistle', density=True, label='Mass Histogram')
            y_bgn = pdf_bgn(bins_bgn[:-1], *m_comb_bgn.values)  # Replace pdf_bgn with your background fitting function

            # Create Bokeh figure
            p = figure(title='Muon Pair Invariant Mass (background)', x_axis_label='Mass (GeV/c^2)', y_axis_label='Candidates')
            p.add_layout(Title(text='Fitted Background Exponential', align='center'), 'above')

            # Plot background histogram
            p.quad(top=n_bgn, bottom=0, left=bins_bgn[:-1], right=bins_bgn[1:], fill_color='thistle', line_color='white')

            # Plot fitted exponential
            x = np.linspace(bins_bgn[0], bins_bgn[-1], 1000)
            y_bgn = pdf_bgn(x, *m_comb_bgn.values)  # Replace pdf_bgn with your background fitting function
            p.line(x, y_bgn, color='darkblue', legend_label='Fitted Background Exponential')
              # Add hover tool to display coordinates
            hover = HoverTool(tooltips=[('Mass', '@left{0.00} - @right{0.00} GeV/c^2'), ('Candidates', '@hist')])
            p.add_tools(hover)


            # Display the plot using Streamlit
            st.bokeh_chart(p)

        else:
            # Create a Matplotlib plot

    
            # Perform histogram
            n_bgn, bins_bgn, patches = plt.hist(bgn['InvMass'], bins=1000, color='thistle', density=True, label='Mass Histogram')
            y_bgn = pdf_bgn(bins_bgn[:-1], *m_comb_bgn.values)  # Replace pdf_bgn with your background fitting function

            # Plot background histogram and fitted exponential
            plt.plot(bins_bgn[:-1], y_bgn, color='darkblue', label='Fitted Background Exponential')
            plt.xlabel('Mass (GeV/c^2)')
            plt.ylabel('Candidates')
            plt.title('Muon Pair Invariant Mass (background)')
            plt.legend()

            # Display the plot
            st.pyplot()

    def pdf_tot2(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3):
    
        """
        pdf function describing the entire mass range of interest. Each of the three Upsilon peaks are described by a 
        gaussian and the background is described by an exponential curve
        
        Free parameters are the Upsilon peak means, and the first peak resolution. The two remaining peak resolutions are related
        to the original peak resolution
        """
        
        a = df['InvMass'].min()
        b = df['InvMass'].max()
        
        tau = m_comb_bgn.values[0]
        s_2 = s_1 * mu_2/mu_1
        s_3 = s_1 * mu_3/mu_1
        
        # one gaussian PDF per mass peak 
        g_1 = (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*s_1**2)) 
        g_2 = (1/(s_2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*s_2**2)) 
        g_3 = (1/(s_3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*s_3**2)) 
        
        # exponential pdf for background    
        exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

        
        return F1*(exponential) + (1 - F1 - F2 - F3)* g_1  + F2*g_2 + F3*g_3

    
    def pdf_tot2_for_plot(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3):
        
        """
        This function is used to provide the signal distribution so that it may be plotted
        """
        
        a = df['InvMass'].min()
        b = df['InvMass'].max()
        
        tau = m_comb_bgn.values[0]
        s_2 = s_1 * mu_2/mu_1
        s_3 = s_1 * mu_3/mu_1
        
        # one gaussian PDF per mass peak 
        g_1 = (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*s_1**2)) 
        g_2 = (1/(s_2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*s_2**2)) 
        g_3 = (1/(s_3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*s_3**2)) 
        
        # exponential pdf for background    
        exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

        
        return (1 - F1 - F2 - F3)* g_1  + F2*g_2 + F3*g_3
    
    
    # combined pdf likelihood
    def likelihood_tot2(F1, F2, F3, s_1, mu_1, mu_2, mu_3):
        
        """
        Likelihood function for the triple gaussian fit 
        """
        
        return -np.sum(np.log(pdf_tot2(df["InvMass"], F1, F2, F3, s_1, mu_1, mu_2, mu_3)))
                        

    #Determine the optimal parameters for the full fit

    m_comb_tot2 = Minuit(likelihood_tot2, F1 = 0.6, F2 = 0.15, F3 = 0.15, s_1 = 0.05, mu_1 = 9.35, mu_2=10, mu_3=10.4)

    m_comb_tot2.migrad()  # run optimiser
    m_comb_tot2.hesse()   # run covariance estimator



    if selected_graph=='Full fit':
        subgraph_options = {'mass histogram, fitted function, and signal distribution','plot the residuals '}
        selected_subgraph=st.sidebar.radio("Select a graph of full fit",subgraph_options)
        if selected_subgraph=='mass histogram, fitted function, and signal distribution':
            
            fig1,ax1=plt.subplots()
            (nfull, binsfull, patchesfull) = ax1.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
            ax1.set_xlabel('Mass (GeV/c^2)')
            ax1.set_ylabel('Candidates')
            ax1.set_title('Triple Gaussian Fit')

            x = binsfull[:-1]

            y_tot2 = pdf_tot2(x, *m_comb_tot2.values)
            ax1.plot(x, y_tot2, color = 'darkblue', label = 'Fitted Function')

            signl = pdf_tot2_for_plot(x, *m_comb_tot2.values)

            ax1.plot(x, signl, color = 'red', linestyle = '--', label = 'Signal Distribution')
            ax1.legend()
            

            st.pyplot(fig1)

        elif selected_subgraph=='plot the residuals ':
            fig1,ax1=plt.subplots()
            (nfull, binsfull, patchesfull) = ax1.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')

            x = binsfull[:-1]

            y_tot2 = pdf_tot2(x, *m_comb_tot2.values)
            plt.plot(x, y_tot2, color = 'darkblue', label = 'Fitted Function')

            signl = pdf_tot2_for_plot(x, *m_comb_tot2.values)
            residuals = y_tot2 - nfull
            plt.plot(binsfull[:-1], residuals, color = 'slateblue', marker = '.', linestyle = 'None')
            plt.axhline(y=0, color='Grey', linestyle='-')
            plt.xlabel('Mass (GeV/c^2)')
            plt.ylabel('Residuals')
            st.pyplot(plt)











# # Monte Carlo Data to find crystall ball params

# File upload section
st.sidebar.title("Monte Carlo")
file2 = st.sidebar.file_uploader("Upload the file", type="bin")

# Graph selection section
st.sidebar.title("Graph Selection")
graph_options2 = ['Data','mass histogram and fitted function','Triple Composite Fit','Ratios of second and third peak yield with respect to first peak yield']
selected_graph2 = st.sidebar.radio("Select a graph", graph_options2)


if file2 is not None:
    data = file2.read()
    datalist = np.frombuffer(data, dtype=np.float32)   
    nevent = int(len(datalist)/6)
    xdata = np.split(datalist,nevent) 

    cols = ["InvMass", "TransMomPair", "Rapid", "MomPair", "TransMom1", "TransMom2"]
    df_mc = pd.DataFrame(xdata, columns = cols)

    mass_min = df_mc['InvMass'].min()
    mass_max = df_mc['InvMass'].max()

    if selected_graph == 'Data':
        if st.sidebar.radio("Show DataFrame", [False, True]):
            st.title("Data Frame")
            st.dataframe(df_mc)
            

    def pdf_cb(x, beta, loc, scale, sigma):
    
        """
        Pdf function describing the first Upsilon peak using a gaussian and crystal function that share a mean
        """
        
        m = 1.0001
        F = 0.75
        
        function = lambda x: stats.crystalball.pdf(x, beta, m, loc, scale)
        area = integrate.quad(function, mass_min, mass_max)[0]
        
        crystal_ball = stats.crystalball.pdf(x, beta, m, loc, scale) / area
        gaussian = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-loc)**2 / (2*sigma**2))  

        
        return F*crystal_ball + (1-F)*gaussian
    
    def like_cb(beta, loc, scale, sigma):
    
        """
        Likelihood function for the monte carlo fit
        """
        
        
        return -np.sum(np.log(pdf_cb(df_mc["InvMass"], beta, loc, scale, sigma)))

    #Determine optimal parameter for the crystal ball + gaussian fit 

    m_comb_mc = Minuit(like_cb, beta = 2, loc = 9.45, scale = 0.04, sigma = 0.02)

    m_comb_mc.migrad()  # run optimiser
    m_comb_mc.hesse()   # run covariance estimator

    if selected_graph2=='mass histogram and fitted function':
        fig,ax=plt.subplots()
        (n_mc, bins_mc, patches_mc) = ax.hist(df_mc['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
        ax.set_xlabel('Mass (GeV/c^2)')
        ax.set_ylabel('Candidates')
        ax.set_title('Muon Pair Invariant Mass')
        ax.plot(bins_mc[:-1], pdf_cb(bins_mc[:-1], *m_comb_mc.values), color = 'darkblue', label = 'Fitted Function')
        ax.legend()
        st.pyplot(fig)

    def pdf_crys_all(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1):
        """
        Pdf function to describe the entire mass distribution. Each of the three upsilon peaks are described by a gaussian and 
        crystal ball function that share a mean. As before, the background is described by the same exponential
        
        As before, the resolutions of the second and third peaks are related to the resolution of the first peak, which is kept 
        floating in the fit 
        """
        a = df['InvMass'].min()
        b = df['InvMass'].max()
        
        tau = m_comb_bgn.values[0]
        s_2 = s_1 * mu_2/mu_1
        s_3 = s_1 * mu_3/mu_1
        
        beta = m_comb_mc.values[0]
        m = 1.0001
        
        function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
        function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
        function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)
        
        area1 = integrate.quad(function1, a, b)[0]
        area2 = integrate.quad(function2, a, b)[0]
        area3 = integrate.quad(function3, a, b)[0]
        
        sigma2 = sigma1 * mu_2/mu_1
        sigma3 = sigma1 * mu_3/mu_1
        
        # one crystall ball PDF per mass peak 
        cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
        g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  
        
        cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
        g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  
        
        cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
        g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  
        
        
        # exponential pdf for background    
        exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

        return F1*(exponential) + F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)

    def pdf_for_plot(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1):
    
        """
        Function used to obtain the signal distribution which is then plotted
        """
        
        a = df['InvMass'].min()
        b = df['InvMass'].max()
        
        tau = m_comb_bgn.values[0]
        s_2 = s_1 * mu_2/mu_1
        s_3 = s_1 * mu_3/mu_1
        
        beta = m_comb_mc.values[0]
        m = 1.0001
        
        function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
        function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
        function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)
        
        area1 = integrate.quad(function1, a, b)[0]
        area2 = integrate.quad(function2, a, b)[0]
        area3 = integrate.quad(function3, a, b)[0]
        
        sigma2 = sigma1 * mu_2/mu_1
        sigma3 = sigma1 * mu_3/mu_1
        
        # one crystall ball PDF per mass peak 
        cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
        g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  
        
        cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
        g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  
        
        cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
        g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  
        
        
        # exponential pdf for background    
        exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

        
        return F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)

    def like_cb(F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1):
    
        """
        Likelihood function for full fit using gaussian and crystal ball for each peak
        """
        
        return -np.sum(np.log(pdf_crys_all(df["InvMass"], F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1)))
    
    #Determine optimal paramaters for the full gaussian+crystal ball fit

    m_comb_crys = Minuit(like_cb, F1 = 0.82, F2 = 0.12, F3 = 0.013, s_1 = 0.04, mu_1 = 9.45, mu_2=10.01, mu_3=10.35, sigma1 = 0.02)

    m_comb_crys.migrad()  # run optimiser
    m_comb_crys.hesse()   # run covariance estimator

    print(m_comb_crys.values)  # print estimated values
    print(m_comb_crys.errors)  

    if selected_graph2=='Triple Composite Fit':

        fig1,ax1=plt.subplots()
        (n_cry, bins_cry, patches_cry) = ax1.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
        ax1.set_xlabel('Mass (GeV/c^2)')
        ax1.set_ylabel('Candidates')
        ax1.set_title('Triple Composite Fit')


        y_tot2_cry = pdf_crys_all(bins_cry[:-1], *m_comb_crys.values)
        ax1.plot(bins_cry[:-1], y_tot2_cry, color = 'darkblue', label = 'Fitted Function')

        signal = pdf_for_plot(bins_cry[:-1], *m_comb_crys.values)
        ax1.plot(bins_cry[:-1], signal , color = 'red', linestyle = '--', label = 'Signal Distribution')
        ax1.legend()
        st.pyplot(fig1)

        fig2,ax2=plt.subplots()
        #Plot the residuals

        residuals_cry = y_tot2_cry - n_cry
        ax2.plot(bins[:-1], residuals_cry, color = 'slateblue', marker = '.', linestyle = 'None')
        ax2.axhline(y=0, color='Grey', linestyle='-')
        ax2.set_xlabel('Mass (GeV/c^2)')
        ax2.set_ylabel('Residuals')
        st.pyplot(fig2)


    sysmu1 = np.abs( m_comb_crys.values[4] - m_comb_tot2.values[4])
    sysmu2 = np.abs( m_comb_crys.values[5] - m_comb_tot2.values[5])
    sysmu3 = np.abs( m_comb_crys.values[6] - m_comb_tot2.values[6])


    totmu1 = np.sqrt( sysmu1**2 + m_comb_crys.errors[4]**2)
    totmu2 = np.sqrt( sysmu2**2 + m_comb_crys.errors[5]**2)
    totmu3 = np.sqrt( sysmu3**2 + m_comb_crys.errors[6]**2)

    #Signal Yeilds

    df_pbin = df[ (df['PseudoRapid']>=2.5) & (df['PseudoRapid']<=5.5) ]
    def bin_anal ( dfpbin ): 
    
        """
        Performs the crystal ball composite fit for 13 bins of transverse momentum in order to compare parameters
        """
        
        param = []
        
        for i in range(13):
        
            df_cuta = dfpbin[ (dfpbin['TransMomPair']>=(0 + i)) & (dfpbin['TransMomPair']<=(1 + i)) ]
            
            def pdf_crys_bina(x, F1, F2, F3, s_1, sigma1):
        
                a = x.min()
                b = x.max()

                mu_1 = m_comb_crys.values[4]
                mu_2 = m_comb_crys.values[5]
                mu_3 = m_comb_crys.values[6]

                tau = m_comb_bgn.values[0]
                s_2 = s_1 * mu_2/mu_1
                s_3 = s_1 * mu_3/mu_1

                beta = m_comb_mc.values[0]
                m = 1.0001

                function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
                function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
                function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)

                area1 = integrate.quad(function1, a, b)[0]
                area2 = integrate.quad(function2, a, b)[0]
                area3 = integrate.quad(function3, a, b)[0]

                sigma2 = sigma1 * mu_2/mu_1
                sigma3 = sigma1 * mu_3/mu_1

                # one crystall ball PDF per mass peak 
                cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
                g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  

                cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
                g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  

                cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
                g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  

                # exponential pdf for background    
                exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

                return F1*(exponential) + F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)


            def like_bina(F1, F2, F3, s_1, sigma1):
        
                return -np.sum(np.log(pdf_crys_bina(df_cuta["InvMass"], F1, F2, F3, s_1, sigma1)))
        
            m_comb_bina = Minuit(like_bina, F1 = 0.82, F2 = 0.12, F3 = 0.013, s_1 = 0.04, sigma1 = 0.02)

            m_comb_bina.migrad()  # run optimiser
            m_comb_bina.hesse()   # run covariance estimator

            param.extend([m_comb_bina.values[0],m_comb_bina.errors[0] , m_comb_bina.values[1],m_comb_bina.errors[1], m_comb_bina.values[2],m_comb_bina.errors[2], m_comb_bina.values[3], m_comb_bina.values[4]])

        return param


    param = np.array (bin_anal ( df_pbin ))


    # create new data frame for transverse momentum bin analysis
    nevent = int(len(param)/8)
    xdata = np.split(param,nevent) 

    cols = ["F1", "eF1", "F2", "eF2", "F3", "eF3", "s_1", "sigma1"]
    frameBIN = pd.DataFrame(xdata, columns = cols)
    

    ratios = []

    for i in range(13):
        
        parameters = np.array(frameBIN.iloc[i].tolist())
        F1 = parameters[0]
        eF1 = parameters[1]
        F2 = parameters[2]
        eF2 = parameters[3]
        F3 = parameters[4]
        eF3 = parameters[5]
        
        F4 = 1 - F1 - F2 - F3
        eF4 = np.sqrt(eF1**2 + eF2**2 + eF3**2)
        
        
        #Errors calculated using propogation of errors
        cros2 = F4/F2
        eC2 =   cros2 * np.sqrt( (eF4/F4)**2 + (eF2/F2)**2 )
        cros3 = F3/F2
        eC3 =   cros3 * np.sqrt( (eF4/F4)**2 + (eF3/F3)**2 )

        ratios.extend([cros2, eC2, cros3, eC3])
        

    ratios = np.array(ratios)

    #Data frame to contain ratio data and errors
    ratio = int(len(ratios)/4)
    ratio_data = np.split(ratios,ratio) 

    cols = ["cros2", "eC2", "cros3", "eC3"]
    dfR = pd.DataFrame(ratio_data, columns = cols)

    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])

    if selected_graph2=='Ratios of second and third peak yield with respect to first peak yield':

        fig1,ax1=plt.subplots()
        ax1.errorbar(x , dfR["cros2"], yerr= dfR["eC2"], color = 'Blue', label = 'Y(2S)/Y(1S)', fmt = '+')
        ax1.errorbar(x , dfR["cros3"], yerr= dfR["eC3"], color = 'fuchsia', label = 'Y(3S)/Y(1S)', fmt = '+')
        ax1.set_xlabel('Transverse Momentum (GeV/c)', fontsize = 12)
        ax1.set_ylabel(r'$R^{iS/1S}$', fontsize = 12)
        ax1.text(6.5, 0.04,'2.5 < ' r'$\eta$' ' < 5.5', fontsize = 14)
        ax1.legend( fontsize = 12)
        st.pyplot(fig1)

    # # Pseudorapidity bins 
    df_pbin2 = df[ (df['TransMomPair']>=0) & (df['TransMomPair']<=15) ]

    def bin_rap ( dfpbin ): 
        
        """
        Performs the crystal ball composite fit for 5 bins of pseudorapidity in order to compare parameters
        """
        
        param = []
        
        for i in range(5):
        
            df_cuta = dfpbin[ (dfpbin['PseudoRapid']>=(2.5 + 0.5 * i)) & (dfpbin['PseudoRapid']<=(3.0 + 0.5*i)) ]
            
            def pdf_crys_bina(x, F1, F2, F3, s_1, sigma1):
        
                a = x.min()
                b = x.max()

                #pr = 1 + 2.5 + 0.5 * i 
                
                mu_1 = m_comb_crys.values[4]
                mu_2 = m_comb_crys.values[5]
                mu_3 = m_comb_crys.values[6]

                tau = m_comb_bgn.values[0]
                #s_1 = pr * s_1
                s_2 = s_1 * mu_2/mu_1
                s_3 = s_1 * mu_3/mu_1

                beta = m_comb_mc.values[0]
                m = 1.0001

                function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
                function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
                function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)

                area1 = integrate.quad(function1, a, b)[0]
                area2 = integrate.quad(function2, a, b)[0]
                area3 = integrate.quad(function3, a, b)[0]

                sigma2 = sigma1 * mu_2/mu_1
                sigma3 = sigma1 * mu_3/mu_1

                # one crystall ball PDF per mass peak 
                cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
                g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  

                cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
                g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  

                cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
                g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  


                # exponential pdf for background    
                exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )


                return F1*(exponential) + F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)


            def like_bina(F1, F2, F3, s_1, sigma1):
        
                return -np.sum(np.log(pdf_crys_bina(df_cuta["InvMass"], F1, F2, F3, s_1, sigma1)))
        
            m_comb_bina = Minuit(like_bina, F1 = 0.82, F2 = 0.12, F3 = 0.013, s_1 = 0.04, sigma1 = 0.02)

            m_comb_bina.migrad()  # run optimiser
            m_comb_bina.hesse()   # run covariance estimator
    

            param.extend([m_comb_bina.values[0], m_comb_bina.errors[0], m_comb_bina.values[1], m_comb_bina.errors[1], m_comb_bina.values[2], m_comb_bina.errors[2], m_comb_bina.values[3], m_comb_bina.errors[3], m_comb_bina.values[4], m_comb_bina.errors[4]])

        return param

    param_rapid = np.array(bin_rap ( df_pbin2 ))
    # create new data frame for pseudorapidity bin analysis
    nevent_rapid = int(len(param_rapid)/10)
    xdata_rapid = np.split(param_rapid,nevent_rapid) 

    cols = ["F1", "error F1",  "F2", "error F2", "F3", "error F3", "s_1", "error s_1", "sigma1", "error sigma1"]
    frameBINrapid = pd.DataFrame(xdata_rapid, columns = cols)


    #Ratios of second and third peak yield with respect to first peak yield

    ratios_rapid = []

    for i in range(5):
        
        parameters = np.array(frameBINrapid.iloc[i].tolist())
        F1 = parameters[0]
        eF1 = parameters[1]
        F2 = parameters[2]
        eF2 = parameters[3]
        F3 = parameters[4]
        eF3 = parameters[5]
        
        #Errors calculated using propogation of errors
        F4 = 1 - F1 - F2 - F3
        eF4 = np.sqrt(eF1**2 + eF2**2 + eF3**2)

        cros2 = F4/F2
        eC2 =   cros2 * np.sqrt( (eF4/F4)**2 + (eF2/F2)**2 )
        
        cros3 = F3/F2
        eC3 = cros3*   np.sqrt( (eF4/F4)**2 + (eF3/F3)**2 )

        ratios_rapid.extend([cros2, eC2, cros3, eC3])
        
        #print(cros2)

    ratios_rapid = np.array(ratios_rapid)



    #data frame to contain ratios and errors
    ratio_rapid = int(len(ratios_rapid)/4)
    ratio_datarapid = np.split(ratios_rapid,ratio_rapid) 

    cols = ["cros2", "errorC2", "cros3", "errorC3"]
    dfR_rapid = pd.DataFrame(ratio_datarapid, columns = cols)

    
    x1 = np.array([2.75, 3.25, 3.75, 4.25, 4.75])
    

    if selected_graph2=='Ratios of second and third peak yield with respect to first peak yield wrt pseudorapidity':

        fig,ax=plt.subplots()
        ax.errorbar(x1 , dfR_rapid["cros2"], yerr= dfR_rapid["errorC2"], color = 'Blue', label = 'Y(2S)/Y(1S)', fmt = '+')
        ax.errorbar(x1 , dfR_rapid["cros3"], yerr= dfR_rapid["errorC3"], color = 'fuchsia', label = 'Y(3S)/Y(1S)', fmt = '+')
        ax.set_xlabel('Pseudorapidity', fontsize = 12)
        ax.set_ylabel(r'$R^{iS/1S}$',fontsize = 12 )
        ax.legend(loc = [0.6, 0.4], fontsize = 12)
        ax.text(4.3, 0.19, r'$p_T$' ' < 15', fontsize = 14)

        st.pyplot(fig)


