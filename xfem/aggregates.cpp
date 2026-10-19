/// from ngsxfem
#include "../xfem/cutinfo.hpp"
#include "../xfem/aggregates.hpp"


#include <unordered_map>

using namespace ngsolve;
using namespace ngfem;

namespace ngcomp
{

  void ElementPatches::Update (Table<size_t> elements)
  {
    patch = std::move (elements);
    TableCreator<size_t> creator (patch.Size ());
    for (; !creator.Done (); creator++)
      for (size_t p : Range (patch.Size ()))
        for (size_t el : patch[p])
          for (int facet : ma->GetElFacets (ElementId (VOL, el)))
            {
              Array<int> neighbours;
              ma->GetFacetElements (facet, neighbours);
              if (neighbours.Size () == 2 && size_t (neighbours[0]) == el
                  && patch[p].Contains (neighbours[1]))
                creator.Add (p, facet);
            }
    patch_facets = creator.MoveTable ();
  }

  PatchAssembly ElementPatches::GetPatchAssembly (size_t patchnr) const
  {
    return { *ma, patch[patchnr], patch_facets[patchnr] };
  }

  void PatchAssembly::GetDofNrs (const FESpace &fes, Array<DofId> &dofs,
                                  bool free_only) const
  {
    dofs.SetSize0 ();
    Array<DofId> element_dofs;
    for (size_t el : elements)
      {
        fes.GetDofNrs (ElementId (VOL, el), element_dofs);
        for (DofId d : element_dofs)
          if (IsRegularDof (d) && !dofs.Contains (d)
              && (!free_only || fes.GetFreeDofs ()->Test (d)))
            dofs.Append (d);
      }
  }

  template <typename SCAL>
  void PatchAssembly::AddMatrix (
      SliceMatrix<SCAL> mat,
      const Array<shared_ptr<BilinearFormIntegrator>> &integrators,
      const FESpace &trial, const FESpace &test,
      FlatArray<DofId> dofs, FlatArray<DofId> dofs_test, LocalHeap &lh) const
  {
    auto add = [&] (FlatMatrix<SCAL> elmat, FlatArray<DofId> cols,
                    FlatArray<DofId> rows) {
      Array<size_t> colmap (cols.Size ()), rowmap (rows.Size ());
      for (size_t j : Range (cols))
        colmap[j] = dofs.Pos (cols[j]);
      for (size_t i : Range (rows))
        rowmap[i] = dofs_test.Pos (rows[i]);
      for (size_t i : Range (rows))
        for (size_t j : Range (cols))
          if (rowmap[i] != Array<DofId>::ILLEGAL_POSITION
              && colmap[j] != Array<DofId>::ILLEGAL_POSITION)
            mat (rowmap[i], colmap[j]) += elmat (i, j);
    };

    Array<shared_ptr<BilinearFormIntegrator>> volume_integrators;
    Array<shared_ptr<FacetBilinearFormIntegrator>> facet_integrators;
    for (auto &bfi : integrators)
      if (auto fbfi = dynamic_pointer_cast<FacetBilinearFormIntegrator> (bfi))
        facet_integrators.Append (fbfi);
      else
        volume_integrators.Append (bfi);

    if (volume_integrators.Size ())
      for (size_t el : elements)
        {
          HeapReset reset (lh);
          ElementId ei (VOL, el);
          Array<DofId> cols, rows;
          trial.GetDofNrs (ei, cols);
          test.GetDofNrs (ei, rows);
          auto &trafo = ma.GetTrafo (ei, lh);
          MixedFiniteElement fel (trial.GetFE (ei, lh), test.GetFE (ei, lh));
          FlatMatrix<SCAL> elmat (rows.Size (), cols.Size (), lh);
          for (auto &bfi : volume_integrators)
            if (bfi->DefinedOnElement (ei.Nr ())
                && bfi->DefinedOn (ma.GetElIndex (ei)))
              {
                auto &mapped = trafo.AddDeformation (bfi->GetDeformation ().get (), lh);
                bfi->CalcElementMatrix (fel, mapped, elmat, lh);
                add (elmat, cols, rows);
              }
        }

    if (facet_integrators.Size ())
      for (size_t facet : facets)
        {
          HeapReset reset (lh);
          Array<int> els;
          ma.GetFacetElements (facet, els);
          ElementId ei (VOL, els[0]), ei2 (VOL, els[1]);
          Array<DofId> cols, rows, cols2, rows2;
          trial.GetDofNrs (ei, cols);
          test.GetDofNrs (ei, rows);
          trial.GetDofNrs (ei2, cols2);
          test.GetDofNrs (ei2, rows2);
          auto &trafo = ma.GetTrafo (ei, lh);
          auto &trafo2 = ma.GetTrafo (ei2, lh);
          MixedFiniteElement fel (trial.GetFE (ei, lh), test.GetFE (ei, lh));
          MixedFiniteElement fel2 (trial.GetFE (ei2, lh), test.GetFE (ei2, lh));
          Array<int> vertices, vertices2;
          vertices = ma.GetElVertices (ei);
          vertices2 = ma.GetElVertices (ei2);
          int facnr = ma.GetElFacets (ei).Pos (facet);
          int facnr2 = ma.GetElFacets (ei2).Pos (facet);
          cols += cols2;
          rows += rows2;
          FlatMatrix<SCAL> elmat (rows.Size (), cols.Size (), lh);
          for (auto &bfi : facet_integrators)
            if (bfi->DefinedOnElement (ei.Nr ())
                && bfi->DefinedOn (ma.GetElIndex (ei)))
              {
                auto &mapped = trafo.AddDeformation (bfi->GetDeformation ().get (), lh);
                auto &mapped2 = trafo2.AddDeformation (bfi->GetDeformation ().get (), lh);
                bfi->CalcFacetMatrix (fel, facnr, mapped, vertices,
                                      fel2, facnr2, mapped2, vertices2, elmat, lh);
                add (elmat, cols, rows);
              }
        }
  }

  template <typename SCAL>
  void PatchAssembly::AddVector (
      FlatVector<SCAL> vec,
      const Array<shared_ptr<LinearFormIntegrator>> integrators[4],
      const FESpace &test, FlatArray<DofId> dofs_test, LocalHeap &lh) const
  {
    for (size_t el : elements)
      {
        HeapReset reset (lh);
        ElementId ei (VOL, el);
        Array<DofId> rows;
        test.GetDofNrs (ei, rows);
        auto &fel = test.GetFE (ei, lh);
        auto &trafo = ma.GetTrafo (ei, lh);
        FlatVector<SCAL> elvec (rows.Size (), lh);
        for (auto vb : { VOL, BND, BBND, BBBND })
          for (auto &lfi : integrators[vb])
            if (lfi->DefinedOnElement (ei.Nr ())
                && lfi->DefinedOn (ma.GetElIndex (ei)))
              {
                auto &mapped = trafo.AddDeformation (lfi->GetDeformation ().get (), lh);
                lfi->CalcElementVector (fel, mapped, elvec, lh);
                for (size_t i : Range (rows))
                  {
                    size_t row = dofs_test.Pos (rows[i]);
                    if (row != Array<DofId>::ILLEGAL_POSITION)
                      vec[row] += elvec[i];
                  }
              }
      }
  }

  template void PatchAssembly::AddMatrix<double> (
      SliceMatrix<double>, const Array<shared_ptr<BilinearFormIntegrator>> &,
      const FESpace &, const FESpace &, FlatArray<DofId>, FlatArray<DofId>,
      LocalHeap &) const;
  template void PatchAssembly::AddMatrix<Complex> (
      SliceMatrix<Complex>, const Array<shared_ptr<BilinearFormIntegrator>> &,
      const FESpace &, const FESpace &, FlatArray<DofId>, FlatArray<DofId>,
      LocalHeap &) const;
  template void PatchAssembly::AddVector<double> (
      FlatVector<double>, const Array<shared_ptr<LinearFormIntegrator>> *,
      const FESpace &, FlatArray<DofId>, LocalHeap &) const;
  template void PatchAssembly::AddVector<Complex> (
      FlatVector<Complex>, const Array<shared_ptr<LinearFormIntegrator>> *,
      const FESpace &, FlatArray<DofId>, LocalHeap &) const;

  ElementAggregation::ElementAggregation (shared_ptr<MeshAccess> ama)
    : ElementPatches(ama)
  {
    //std::cout << "Hello from element aggregation" << endl;
  }

  void ElementAggregation::Update(shared_ptr<BitArray> & root_els, shared_ptr<BitArray> & bad_els, LocalHeap & lh)
  {
    const size_t ne = ma->GetNE();
    const size_t nf = ma->GetNFacets();

    el_roots = root_els;

    if (ma->GetCommunicator().Size() > 1)
      throw Exception("ElementAggregation::Update:: No Aggregation implementation for MPI yet");

    /*
    BitArray fine_facet(nf);
    fine_facet.Clear();
    IterateRange
      (ma->GetNE(VOL), lh,
       [&] (int elnr, LocalHeap & lh)
       {
         Array<int> fanums(0,lh);
         fanums = ma->GetElFacets (ElementId(VOL,elnr));
         for (int j=0; j<fanums.Size(); j++)
           fine_facet.SetBitAtomic(fanums[j]);
       });
    */


    shared_ptr<BitArray> facets = GetFacetsWithNeighborTypes(ma,root_els,bad_els,false,false,true,lh);
    shared_ptr<BitArray>patch_root_elements = make_shared<BitArray>(ne);
    patch_root_elements->Clear();
    //cout << *patch_root_elements << endl;
    *patch_root_elements |= *root_els;
    //cout << *patch_root_elements << endl;
    *patch_root_elements &= *GetElementsWithNeighborFacets(ma,facets,lh);
    //cout << *patch_root_elements << endl;
    size_t nc = 0;
    for (size_t i : Range(ne)){
      if (patch_root_elements->Test(i)){
        nc ++;
      }
    }
    (*testout) << "Number of (nontrivial) patches: " << nc << endl;

    element_to_patch.SetSize(ne);
    element_to_patch = -1;

    facet_to_patch.SetSize(nf);
    facet_to_patch = -1;

    patch_roots.SetSize(nc);
    size_t counter = 0;
    for (size_t i: Range(ma->GetNE())){
      if (patch_root_elements->Test(i)){
        element_to_patch[i] = counter;
        patch_roots[counter]=i;
        counter++;
      }
      else
        element_to_patch[i] = -1;
    }

    BitArray front_elements(ne);
    BitArray new_front_elements(ne);
    new_front_elements = *patch_root_elements;

    BitArray not_covered_yet(ne);
    not_covered_yet.Clear();
    not_covered_yet |= *bad_els;
    BitArray newly_covered(ne);
    bool uncovered_elements_left = true;

    inner_patch_facets = make_shared<BitArray> (nf);
    inner_patch_facets->Clear();

    while (not_covered_yet.NumSet() > 0)
    {
      front_elements = new_front_elements;
      new_front_elements.Clear();
      newly_covered.Clear();
      //IterateRange(ne, lh, [&] (int elnr, LocalHeap & lh)
      for (size_t elnr = 0; elnr < ne; elnr++)
      {
        size_t patch_id = element_to_patch[elnr];
        if (front_elements.Test(elnr))
        {
          Array<int> fanums(0,lh);
          fanums = ma->GetElFacets (ElementId(VOL,elnr));
          for (int facnr : fanums)
          {
            Array<int> elnums(0,lh);
            ma->GetFacetElements (facnr, elnums);
            for (int nelnr : elnums)
            {
              if (nelnr == elnr)
                continue; // only consider the neighbor elements
              if (not_covered_yet.Test(nelnr) && (!newly_covered.Test(nelnr)))
              {
                inner_patch_facets->SetBitAtomic(facnr);
                facet_to_patch[facnr]=patch_id;
                new_front_elements.SetBitAtomic(nelnr);
                element_to_patch[nelnr] = patch_id;
                newly_covered.SetBitAtomic(nelnr);
              }
              else
              {
                if (element_to_patch[elnr] == element_to_patch[nelnr])
                {
                  inner_patch_facets->SetBitAtomic(facnr);
                  facet_to_patch[facnr]=patch_id;
                }
              }
            }
          }

        }
      }
      not_covered_yet &= ~newly_covered;

      // Guard against non-termination: if a sweep covers no new element while
      // bad elements remain uncovered, those elements belong to a connected
      // component that contains no root element. They can never be aggregated,
      // so the while-loop would spin forever. Fail loudly instead of hanging.
      if (newly_covered.NumSet() == 0 && not_covered_yet.NumSet() > 0)
        throw Exception("ElementAggregation::Update: " + ToString(not_covered_yet.NumSet()) +
                        " bad element(s) are not connected (via facets) to any root element. "
                        "Each connected component of the aggregation region must contain at "
                        "least one root element; otherwise these elements cannot be aggregated. "
                        "Check the root/bad element marking (e.g. a level set that is no longer "
                        "a signed-distance function can create extension-band islands detached "
                        "from the interior).");
    }

    (*testout) << "element_to_patch = \n" << element_to_patch << endl;

    // Keep a row for every root, including roots that received no leaf.
    TableCreator<size_t> creator(nc);
    for ( ; !creator.Done(); creator++)
      for (auto elnr : Range(ne))
      {
        if (element_to_patch[elnr] >= 0 && patch_roots[element_to_patch[elnr]] != elnr)
          creator.Add (element_to_patch[elnr], elnr);
      }
    patch_leafs = creator.MoveTable();

    TableCreator<size_t> pcreator(root_els->NumSet());
    for ( ; !pcreator.Done(); pcreator++)
      {
        for (auto pnr : Range(nc))
          {
            // Extension embedding keeps the root DOFs first.
            pcreator.Add(pnr, patch_roots[pnr]);
            for (size_t el : patch_leafs[pnr])
              pcreator.Add(pnr, el);
          }
        size_t pnr = nc;
        for (auto el : Range(ne))
          if (root_els->Test(el) && element_to_patch[el] < 0)
            pcreator.Add(pnr++, el);
      }
    ElementPatches::Update(pcreator.MoveTable());

    (*testout) << "patch id to included element: \n " << patch << endl;
    (*testout) << "patch id to supporting element: \n " << patch_roots << endl;
    (*testout) << "patch id to supported elements: \n " << patch_leafs << endl;
    (*testout) << "patch id to supported connecting facets: \n " << patch_facets << endl;
    (*testout) << "inner patch facets: \n " << *inner_patch_facets << endl;

    n_els_in_nontrivial_patches = 0;
    n_nontrivial_patches = nc;

    n_trivial_els = root_els->NumSet() - n_nontrivial_patches;

    el_in_nontrivial_patch = make_shared<BitArray>(ne);
    el_in_nontrivial_patch->Clear();
    for (size_t i: Range(nc))
    {
      n_els_in_nontrivial_patches ++;
      el_in_nontrivial_patch->SetBitAtomic(patch_roots[i]);
      for (size_t j : patch_leafs[i])
      {
        n_els_in_nontrivial_patches ++;
        el_in_nontrivial_patch->SetBitAtomic(j);
      }
    }

    el_in_trivial_patch = make_shared<BitArray>(ne);
    el_in_trivial_patch->Clear();

    *el_in_trivial_patch |= *root_els;
    *el_in_trivial_patch &= ~(*el_in_nontrivial_patch);

    
  }

  template <typename TFUNC>
  void PatchLoop (shared_ptr<ElementAggregation> elagg, 
                  bool include_trivial_patches,
                  shared_ptr<FESpace> fes_trial,
                  shared_ptr<FESpace> fes_test,
                  shared_ptr<SumOfIntegrals> bf,
                  shared_ptr<SumOfIntegrals> lf,
                  LocalHeap & clh,
                  const TFUNC & patchwise_func
                  )
  {
    size_t npatch = elagg->GetNPatches(include_trivial_patches);
    if (fes_trial->GetFreeDofs()->NumSet() < fes_trial->GetNDof())
      throw Exception("cannot handle non-trivial freedofs array");

    Array<shared_ptr<BilinearFormIntegrator>> bfis;
    Array<shared_ptr<LinearFormIntegrator>> lfis[4];
    if (bf)
      for (auto icf : bf->icfs)
        if (icf->dx.vb == VOL)
          bfis.Append(icf->MakeBilinearFormIntegrator());
    if (lf)
      for (auto icf : lf->icfs)
        if (icf->dx.vb == VOL)
          lfis[VOL].Append(icf->MakeLinearFormIntegrator());

    ParallelForRange (Range(npatch), [&] (IntRange range)
    {
      LocalHeap lh = clh.Split();
      Array<DofId> patchdofs_trial, patchdofs_test;
      for (size_t p : range)
        {
          HeapReset reset(lh);
          auto local = elagg->GetPatchAssembly(p);
          local.GetDofNrs(*fes_trial, patchdofs_trial);
          local.GetDofNrs(*fes_test, patchdofs_test, true);
          FlatMatrix<> patchmat(patchdofs_test.Size(), patchdofs_trial.Size(), lh);
          FlatVector<> patchvec(patchdofs_test.Size(), lh);
          patchmat = 0.0;
          patchvec = 0.0;
          local.AddMatrix<double>(patchmat, bfis, *fes_trial, *fes_test,
                                  patchdofs_trial, patchdofs_test, lh);
          local.AddVector<double>(patchvec, lfis, *fes_test, patchdofs_test, lh);
          patchwise_func(p, patchmat, patchvec, patchdofs_trial, patchdofs_test, lh);
        }
    });
  }

  void PatchwiseSolve(shared_ptr<ElementAggregation> elagg, 
                      shared_ptr<FESpace> fes,
                      shared_ptr<SumOfIntegrals> bf,
                      shared_ptr<SumOfIntegrals> lf,
                      shared_ptr<BaseVector> vec,
                      LocalHeap & clh
                      )
  {
    size_t n_patch = elagg->GetNPatches(true);
    Array<int> dof_in_npatches(fes->GetNDof());
    dof_in_npatches = 0;
    for (int pi : Range(n_patch))
    {
      Array<DofId> patchdofs;
      elagg->GetPatchAssembly(pi).GetDofNrs(*fes, patchdofs);
      for (DofId d : patchdofs)
        dof_in_npatches[d]++;
    }
    //cout << " - dof_in_npatches - " << dof_in_npatches << endl;
    (*vec) = 0.0;
    PatchLoop(elagg, true, fes, fes, bf, lf, clh, [&] (int p, 
                                                       FlatMatrix<> patchmat, 
                                                       FlatVector<> patchvec,
                                                       Array<DofId> & patchdofs,
                                                       Array<DofId> & patchdofs_dummy,
                                                       LocalHeap & lh
                                                       )
    {
      FlatMatrix<> patchinv(patchdofs.Size(),patchdofs.Size(),lh); 
      CalcInverse(patchmat, patchinv);
      FlatVector<> patchsol(patchdofs.Size(),lh); 
      patchsol = patchinv * patchvec;
      vec->AddIndirect (patchdofs, patchsol);
    });

    ParallelForRange (dof_in_npatches.Size(), [&] (IntRange r)
    {
      //VectorMem<10,SCAL> fluxi(dim);
      VectorMem<10,double> fluxi(1);
      ArrayMem<int,1> dnums(1);
      for (auto i : r)
        if (dof_in_npatches[i] > 1)
        {
          dnums[0] = i;
          vec->GetIndirect (dnums, fluxi);
          fluxi /= double (dof_in_npatches[i]);
          vec->SetIndirect (dnums, fluxi);
        }
    });
    
  }

  shared_ptr<SparseMatrix<double>> SetupExtensionEmbedding(shared_ptr<ElementAggregation> elagg,
                                                           shared_ptr<FESpace> fes,
                                                           shared_ptr<SumOfIntegrals> bf,
                                                           LocalHeap &clh)
  {

    // Setup of Sparse-Matrix for Embedding
    const BitArray & freedofs = *fes->GetFreeDofs();
    shared_ptr<MeshAccess> ma = elagg->GetMesh();

    BitArray non_trivial_dofs(fes->GetNDof());
    non_trivial_dofs.Clear();
    BitArray trivial_dofs(fes->GetNDof());
    trivial_dofs.Clear();
    BitArray root_dofs(fes->GetNDof());
    root_dofs.Clear();

    Array<int> dof_in_npatches(fes->GetNDof());
    dof_in_npatches = 0;

    (*testout) << "dof_in_npatches:\n" << dof_in_npatches << endl;

    ma->IterateElements(VOL, clh, [&](auto ei, LocalHeap &mlh)
    {
      Array<DofId> dofs;
      if ( (*elagg->GetElsInNontrivialPatch())[ei.Nr()] )
      {
        const bool root = (*elagg->GetRootElements())[ei.Nr()];
        fes->GetDofNrs(ei, dofs);
        for( auto i : dofs )
        {
          non_trivial_dofs.SetBitAtomic(i);
          if (root)
            root_dofs.SetBitAtomic(i);
        }
      }
      if ( (*elagg->GetElsInTrivialPatch())[ei.Nr()] )
      {
        fes->GetDofNrs(ei, dofs);
        for( auto i : dofs )
          trivial_dofs.SetBitAtomic(i);
      }

    });

    BitArray rdofs(fes->GetNDof());
    rdofs.Clear();
    rdofs |= trivial_dofs;
    rdofs |= root_dofs;
    const int nrdofs = rdofs.NumSet(); 
    const size_t n_root_dof = root_dofs.NumSet();
    (*testout) << "n_root_dof: " << n_root_dof << endl;

    size_t n_nontriv_patch = elagg->GetNPatches(false);

    for (int pi : Range(n_nontriv_patch))
    {
      Array<DofId> patchdofs;
      elagg->GetPatchAssembly(pi).GetDofNrs(*fes, patchdofs);
      for (DofId d : patchdofs)
        if (non_trivial_dofs[d] && !root_dofs[d])
          dof_in_npatches[d]++;
    }

    Array<DofId> dof2rdof(fes->GetNDof());
    dof2rdof = -1;
    Array<DofId> rdof2dof(nrdofs);
    rdof2dof = -1;

    int cnt = 0;
    for (int i : Range(fes->GetNDof()))
    {
        if (!non_trivial_dofs.Test(i) || root_dofs.Test(i))
        {
        dof2rdof[i] = cnt;
        rdof2dof[cnt] = i;
        cnt++;
        }
    }

    (*testout) << "dof2rdof \n"
                << dof2rdof << endl;
    (*testout) << "rdof2dof \n"
                << rdof2dof << endl;

    size_t npatch = elagg->GetNPatches(true);
    Table<int> table, rtable;
    TableCreator<int> creator(npatch);
    TableCreator<int> creator2(npatch);
    {
      Array<DofId> patchdofs;
      for (; !creator.Done(); creator++, creator2++)
        for (int pi : Range(npatch))
          {
            elagg->GetPatchAssembly(pi).GetDofNrs(*fes, patchdofs);
            for (DofId d : patchdofs)
              {
                creator.Add(pi,d);
                const int rdof = dof2rdof[d];
                if (IsRegularDof(rdof))
                  creator2.Add(pi,rdof);
              }
          }
      table = creator.MoveTable ();
      rtable = creator2.MoveTable ();
    }
    shared_ptr<SparseMatrix<double>> P
      = make_shared<SparseMatrix<double>> (fes->GetNDof (), 
                                           nrdofs, table, rtable, false);
    
    P->SetZero ();

    double one = 1;
    FlatMatrix<> I(1,1,&one);
    Array<DofId> cdof(1), crdof(1);
    for (int i : Range(fes->GetNDof()))
    {
      if (!non_trivial_dofs[i] || root_dofs[i])
      {
        cdof[0] = i; 
        crdof[0] = dof2rdof[i];
        P->AddElementMatrix(cdof, crdof, I);
      }
    }

    (*testout) << "non_trivial_dofs: " << non_trivial_dofs << endl;
    (*testout) << "P: \n" << *P << endl;

    PatchLoop(elagg, false, fes, fes, bf, nullptr, clh, [&] (int p, 
                                                            FlatMatrix<> patchmat, 
                                                            FlatVector<> patchvec,
                                                            Array<DofId> & patchdofs,
                                                            Array<DofId> & patchdofs_dummy,
                                                            LocalHeap & lh
                                                            )
    {
      Array<size_t> els_in_patch;
      elagg->GetPatch(p, els_in_patch);

      Array<DofId> dofs;
      fes->GetDofNrs(ElementId(VOL, els_in_patch[0]), dofs);
      int nr_dofs_inner = dofs.Size();

      /*int nr_dofs_inner = 0;
      for (auto d : dofs)
        if (freedofs.Test(d))
            nr_dofs_inner++;*/

      (*testout) << "nr dofs root " << nr_dofs_inner << endl;

      int nr_dofs_outer = patchdofs.Size() - nr_dofs_inner;

      FlatMatrix<double> Soo(nr_dofs_outer, nr_dofs_outer, lh);
      Soo = patchmat.Rows(nr_dofs_inner, patchdofs.Size()).Cols(nr_dofs_inner, patchdofs.Size());

      FlatMatrix<double> Soi(nr_dofs_outer, nr_dofs_inner, lh);
      Soi = patchmat.Rows(nr_dofs_inner, patchdofs.Size()).Cols(0, nr_dofs_inner);

      FlatMatrix<double> Sooinv(nr_dofs_outer, nr_dofs_outer, lh);
      CalcInverse(Soo, Sooinv);
      FlatMatrix<double> Q(nr_dofs_outer, nr_dofs_inner, lh);
      Q = -Sooinv * Soi;

      for (int i : Range(nr_dofs_outer))
      {
        Q.Row(i) /= dof_in_npatches[patchdofs[nr_dofs_inner+i]];
      }
      Array<DofId> patchrdofs(nr_dofs_inner);
      for (int i : Range(nr_dofs_inner))
        patchrdofs[i] = dof2rdof[patchdofs[i]];
      (*testout) << "p = " << p << "\n patchdofs: \n" << patchdofs << "\ninner dofs: \n" << patchrdofs << endl;
      P->AddElementMatrix(patchdofs.Range(nr_dofs_inner,patchdofs.Size()), patchrdofs, Q);

      (*testout) << "patch nr " << p << endl;
      // (*testout) << "dofs test\n" << patchdofs_test << endl;
      // (*testout) << "patch vector \n" << patchvec << endl;
      (*testout) << "patch matrix \n"
                 << patchmat << endl;
      (*testout) << "patch dofs\n"
                 << patchdofs << endl;
      // (*testout) << "dofs of root \n" << dofs << endl;
      (*testout) << "Q: \n"
                 << Q << endl;

    });

    // Finalizing steps
    (*testout) << "P result: \n"
               << *P << endl;  
    return P;
  } // end of SetupAgg..
}
